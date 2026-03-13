import logging
import xml.etree.ElementTree as ET
import argparse
import os
import re
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections, utility
import time
from nltk.tokenize import sent_tokenize
import psutil
import io
import sys
import gc
import torch 
import numpy as np

# Set the console to use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Configure logging
logger = logging.getLogger("CandlekeepLogger")
logger.setLevel(logging.DEBUG)  # Change from INFO to DEBUG

# File handler for logging to a file
file_handler = logging.FileHandler("candlekeep_processing.log", mode="a", encoding="utf-8")  # Use UTF-8 encoding
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# Console handler for real-time monitoring
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(console_handler)

def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"[{stage}] Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")

# Function to clean Wikipedia text
def clean_wikipedia_text(raw_text):
    # Step 1: Remove image file tags and their descriptions, including nested brackets
    cleaned_text = re.sub(
        r'\[\[File:[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]', 
        '', 
        raw_text, 
        flags=re.DOTALL
    )

    # Step 2: Remove HTML-like tags (e.g., <ref>, <span>)
    cleaned_text = re.sub(r"<.*?>", "", cleaned_text)

    # Step 3: Replace HTML entities and escaped characters
    html_replacements = {
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&amp;": "&",
        "&apos;": "'",
        "&ndash;": "–",
        "&mdash;": "—",
        "&hellip;": "…",
        "&ldquo;": "“",
        "&rdquo;": "”",
        "&lsquo;": "‘",
        "&rsquo;": "’",
        "&nbsp;": " ",
        "&copy;": "©",
        "&reg;": "®",
        "&euro;": "€",
        "&pound;": "£",
        "&yen;": "¥",
        "&cent;": "¢",
        "&plusmn;": "±",
        "&times;": "×",
        "&divide;": "÷",
        "&eacute;": "é",
        "&agrave;": "à",
        "&uuml;": "ü",
        "&ntilde;": "ñ",
    }
    for entity, replacement in html_replacements.items():
        cleaned_text = cleaned_text.replace(entity, replacement)

    # Step 4: Remove all templates and references, including nested ones
    cleaned_text = re.sub(r"\{\{(?:[^{}]|\{[^{}]*\})*\}\}", "", cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"\{\{Refn\|.*?\}\}", "", cleaned_text, flags=re.DOTALL)  # Remove Refn templates
    cleaned_text = re.sub(r"\{\{[Rr]ef\|.*?\}\}", "", cleaned_text, flags=re.DOTALL)  # Remove Ref templates
    cleaned_text = re.sub(r"\{\{[^{}]*\}\}", "", cleaned_text)  # Catch any remaining simple templates

    # Step 5: Convert Wikipedia hyperlink formatting
    cleaned_text = re.sub(r"\[\[([^\|\]]*?)\|([^\]]+?)\]\]", r"\2", cleaned_text)  # [[linked text|display text]] → display text
    cleaned_text = re.sub(r"\[\[([^\]]+?)\]\]", r"\1", cleaned_text)  # [[linked text]] → linked text

    # Step 6: Remove tables
    cleaned_text = re.sub(r"{\|.*?\|}", "", cleaned_text, flags=re.DOTALL)  # Remove entire table block

    # Step 7: Remove Wikipedia italics (double apostrophes)
    cleaned_text = re.sub(r"''", "", cleaned_text)  # Remove '' for italics

    # Step 8: Normalize remaining Unicode characters
    cleaned_text = re.sub(r"[^\x00-\x7F]+", "", cleaned_text)  # Remove non-ASCII characters if necessary

    # Step 9: Remove extra whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text

# create local url
def construct_kiwix_url(article_title):
    # Normalize the title (spaces to underscores)
    normalized_title = article_title.replace(" ", "_")

    # Construct the Kiwix URL with the fixed /A/ identifier
    kiwix_url = f"http://candlekeep.local/viewer#wikipedia_en_all_maxi_2024-01/A/{normalized_title}"

    return kiwix_url

# Step 1: Extract templates selectively based on a whitelist
def extract_whitelisted_templates(text):
    # Stop processing at the first section header
    intro_end = re.search(r"(={2,})\s*[^=]+?\s*\1", text, re.MULTILINE)
    intro_text = text[:intro_end.start()] if intro_end else text

    # Match all templates in the intro text
    pattern = r"\{\{(.*?)\}\}"
    matches = re.findall(pattern, intro_text)
    logger.debug(f"Extracted templates from intro: {matches}")

    # Define both quality and skip templates
    quality_templates = [
        "featured article", 
        "vital article", 
        "good article", 
        "featured list", 
        "a-class", 
        "ga-class"
    ]
    skip_tags = [
        "stub", 
        "portal", 
        "list", 
        "list-class", 
        "disambiguation"
    ]

    # Initialize return values
    article_quality = None
    skip_article = False
    
    # Process each template
    for match in matches:
        # Extract base template name (before any | character)
        base_template = match.split('|')[0].lower().strip()
        
        # Check for quality templates first
        for quality in quality_templates:
            if quality == base_template:
                article_quality = quality.capitalize()
                logger.debug(f"Article quality determined: {article_quality}")
                break
        
        # Then check for skip tags
        for skip_tag in skip_tags:
            if skip_tag == base_template:
                skip_article = True
                logger.info(f"Skipping article due to template: {skip_tag}")
                break
        
        if skip_article:
            break

    return matches, article_quality, skip_article

# Step 2: Stream the XML and process articles one <page> at a time
def stream_xml(file_path):
    logger.info("Starting XML streaming...")
    log_memory_usage("Before XML Streaming")

    context = ET.iterparse(file_path, events=("start", "end"))
    event, root = next(context)  # Get the root element

    # Extract the namespace from the root element
    namespace = {"ns": root.tag.split("}")[0].strip("{")}

    for event, elem in context:
        if event == "end" and elem.tag == f"{{{namespace['ns']}}}page":
            title_elem = elem.find("ns:title", namespace)
            if title_elem is not None:
                title = title_elem.text
                logger.info(f"Processing <page> element: {title}")
            else:
                logger.warning("No <title> element found in <page>")
                root.clear()
                continue

            # Check for <redirect> tag
            redirect_elem = elem.find("ns:redirect", namespace)
            if redirect_elem is not None:
                logger.info(f"Skipping redirect page: {title}")
                root.clear()
                continue

            # Process the <revision>/<text> content
            revision_elem = elem.find("ns:revision", namespace)
            if revision_elem is not None:
                # Extract the <text> tag and its attributes
                text_elem = revision_elem.find("ns:text", namespace)
                bytes_attr = text_elem.attrib.get("bytes") if text_elem is not None else None
                sha1_attr = text_elem.attrib.get("sha1") if text_elem is not None else None
                content = text_elem.text if text_elem is not None else ""

                # Extract the <timestamp> field
                timestamp_elem = revision_elem.find("ns:timestamp", namespace)
                edit_date = timestamp_elem.text if timestamp_elem is not None else None

                # Skip articles with bytes < 250
                if bytes_attr is not None and int(bytes_attr) < 250:
                    logger.info(f"Skipping article '{title}' due to low byte count: {bytes_attr}")
                    root.clear()
                    continue
            else:
                logger.warning(f"No <revision> element found in <page>: {title}")
                root.clear()
                continue

            # Extract templates from the content
            templates, article_quality, skip_article = extract_whitelisted_templates(content)

            if skip_article:
                logger.info(f"Skipping article '{title}' due to skip tags in templates: {templates}")
                root.clear()
                continue

            # Yield the extracted data
            yield title, content, sha1_attr, edit_date, templates, article_quality

            root.clear()
            log_memory_usage("After Processing Page")
            gc.collect()

def discard_post_see_also(content):
    # Define section markers as regex patterns
    section_markers = [
        r'==\s*See also\s*==',
        r'==\s*References\s*==',
        r'==\s*Citations\s*==',
        r'==\s*External links\s*==',
        r'==\s*Further reading\s*==',
        r'==\s*Notes\s*=='
    ]
    
    # Find the first matching section using regex
    cutoff_pos = len(content)
    for pattern in section_markers:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            pos = match.start()
            if pos < cutoff_pos:
                cutoff_pos = pos
                logger.debug(f"Found cutoff section '{pattern}' at position {pos}")
    
    # If a cutoff point was found, return everything before it
    if cutoff_pos < len(content):
        truncated_content = content[:cutoff_pos].strip()
        logger.debug(f"Content truncated. Last 100 chars: {truncated_content[-100:]}")
        return truncated_content
    
    return content

def extract_sections(content):
    sections = []
    
    # Find all section headers with any number of = symbols (2 or more)
    section_matches = list(re.finditer(r'(={2,})\s*([^=]+?)\s*\1', content))
    
    # Handle articles with no sections - treat entire content as introduction
    if not section_matches:
        logger.debug("No sections found - treating entire content as introduction")
        sections.append({
            "section_hierarchy": "Introduction",
            "content": content.strip(),
            "level": 1
        })
        return sections
    
    # Handle introduction (content before first section)
    intro_content = content[:section_matches[0].start()].strip()
    if intro_content:
        sections.append({
            "section_hierarchy": "Introduction",
            "content": intro_content,
            "level": 1
        })
    
    # Process each section
    for i, match in enumerate(section_matches):
        level = len(match.group(1))  # Count number of = symbols
        section_title = match.group(2).strip()
        
        # Get section content (from end of current section header to start of next section of same or higher level)
        content_start = match.end()
        content_end = len(content)
        
        # Look for the next section that's at the same level or higher in hierarchy
        for next_match in section_matches[i + 1:]:
            next_level = len(next_match.group(1))
            if next_level <= level:  # Found a section at same or higher level
                content_end = next_match.start()
                break
        
        section_content = content[content_start:content_end].strip()
        
        if section_content:  # Only add sections with content
            sections.append({
                "section_hierarchy": section_title,
                "content": section_content,
                "level": level
            })
            logger.debug(f"Found level {level} section: {section_title}")
    
    # Enhanced logging of section hierarchy
    logger.debug("Section hierarchy:")
    for section in sections:
        indent = "  " * (section['level'] - 1) if 'level' in section else ""
        logger.debug(f"{indent}Level {section.get('level', 1)}: {section['section_hierarchy']}")
    
    return sections

# Step 4: Chunk the article with accurate metadata
def chunk_article_with_metadata(article_title, section_data, token_limit, tokenizer, kiwix_url, sha1, article_quality, article_id):
    chunks = []
    current_chunk = []
    current_token_count = 0

    for section in section_data:
        section_title = section["section_hierarchy"]
        paragraphs = section["content"].split("\n\n")

        for i, paragraph in enumerate(paragraphs):
            sentences = sent_tokenize(paragraph)
            
            # Special handling for the last paragraph of the last section
            is_last_paragraph = (section == section_data[-1] and i == len(paragraphs) - 1)

            for j, sentence in enumerate(sentences):
                tokenized = tokenizer(sentence, add_special_tokens=False)
                token_count = len(tokenized['input_ids'])

                # Handle sentences that exceed token limit
                if token_count > token_limit:
                    logger.warning(f"Long sentence found ({token_count} tokens), splitting into smaller chunks")
                    for k in range(0, token_count, token_limit):
                        chunk_tokens = tokenized['input_ids'][k:k + token_limit]
                        chunk_sentence = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                        
                        if current_token_count + len(chunk_tokens) <= token_limit:
                            current_chunk.append(chunk_sentence)
                            current_token_count += len(chunk_tokens)
                        else:
                            # Finalize current chunk
                            finalize_chunk(chunks, current_chunk, article_id, article_title, section_title, kiwix_url, sha1, article_quality)
                            current_chunk = [chunk_sentence]
                            current_token_count = len(chunk_tokens)

                # Check if adding sentence would exceed limit
                elif current_token_count + token_count <= token_limit:
                    current_chunk.append(sentence)
                    current_token_count += token_count
                else:
                    # Normal chunk finalization - complete the current chunk before this sentence
                    finalize_chunk(chunks, current_chunk, article_id, article_title, section_title, kiwix_url, sha1, article_quality)
                    # Start new chunk with this sentence
                    current_chunk = [sentence]
                    current_token_count = token_count

            # Finalize chunk at paragraph boundary unless it's the last paragraph
            if current_chunk and not is_last_paragraph:
                finalize_chunk(chunks, current_chunk, article_id, article_title, section_title, kiwix_url, sha1, article_quality)
                current_chunk = []
                current_token_count = 0

    # Handle any remaining content as its own chunk if it exists and fits
    if current_chunk:
        finalize_chunk(chunks, current_chunk, article_id, article_title, section_title, kiwix_url, sha1, article_quality)

    return chunks

def create_metadata(article_id, chunk_id, article_title, section_title, kiwix_url, sha1, article_quality):
    return {
        "chunk_id": str(chunk_id),  # chunk_id still contains article_id as prefix
        "article_id": int(article_id),  # Store as plain integer
        "article_title": article_title,
        "section_title": section_title,
        "kiwix_url": kiwix_url,
        "sha1_text": sha1,
        "article_quality": article_quality if article_quality else ""
    }

def finalize_chunk(chunks, current_chunk, article_id, article_title, section_title, kiwix_url, sha1, article_quality):
    if current_chunk:
        chunk_id = generate_chunk_id(article_id)
        chunks.append({
            "content": " ".join(current_chunk),
            "metadata": create_metadata(
                article_id,
                chunk_id,
                article_title,
                section_title,
                kiwix_url,
                sha1,
                article_quality
            )
        })

# Step 5: Embed the chunks
def embed_chunks(chunks, model_name):
    # Removed imports: sentence_transformers and torch (moved to top)
    
    # Ensure XPU is available
    if not torch.xpu.is_available():
        logger.error("No XPU available for embedding. Exiting.")
        exit(1)
    
    # Use XPU for embedding
    device = "xpu"
    logger.info(f"Using device: {device}")
    model = SentenceTransformer(model_name, device=device)

    # Process chunks in batches
    chunk_texts = [chunk["content"] for chunk in chunks]
    start_time = time.time()
    batch_size = 64
    embeddings = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        embeddings.extend(model.encode(batch, convert_to_tensor=True, device=device))
    end_time = time.time()

    logger.info(f"Embedding Time (seconds): {end_time - start_time}")
    return embeddings

# Step 6: Insert chunks and embeddings into Milvus
def insert_into_milvus(chunks, embeddings, collection_name, edit_date):
    try:
        # Safety check - don't proceed if there are no chunks
        if not chunks:
            logger.warning("No chunks to insert, skipping Milvus insertion")
            return
            
        connections.connect("default", host="localhost", port="19530")
        
        if not utility.has_collection(collection_name):
            logger.error(f"Collection '{collection_name}' does not exist.")
            exit(1)
        
        collection = Collection(collection_name)
        
        # Convert embeddings to list if it's a tensor
        embedding_list = embeddings if isinstance(embeddings, list) else embeddings.tolist()
        
        # Create lists in exact schema order
        data = [
            [chunk["metadata"]["chunk_id"][:128] for chunk in chunks],  # chunk_id (primary key)
            [chunk["metadata"]["article_id"] for chunk in chunks],  # article_id as INT64
            [chunk["metadata"]["article_title"][:255] for chunk in chunks],  # article_title
            [chunk["metadata"]["section_title"][:255] for chunk in chunks],  # section_title
            [chunk["metadata"]["kiwix_url"][:512] for chunk in chunks],  # kiwix_url
            [chunk["metadata"]["sha1_text"][:128] for chunk in chunks],  # sha1_text
            [chunk["metadata"]["article_quality"][:128] for chunk in chunks],  # article_quality
            [edit_date[:64] for _ in chunks],  # timestamp
            [chunk["content"][:4096] for chunk in chunks],  # content
            embedding_list  # embedding (FLOAT_VECTOR)
        ]
        
        # Debug log metadata values for first chunk
        if chunks:
            logger.debug("First chunk metadata:")
            logger.debug(f"  chunk_id: {chunks[0]['metadata']['chunk_id']}")
            logger.debug(f"  article_id: {chunks[0]['metadata']['article_id']} (type: {type(chunks[0]['metadata']['article_id'])})")
            logger.debug(f"  article_title: {chunks[0]['metadata']['article_title']}")
            logger.debug(f"  section_title: {chunks[0]['metadata']['section_title']}")
        
        start_time = time.time()
        insert_result = collection.insert(data)
        end_time = time.time()
        
        logger.info(f"Inserted {len(data[0])} entities into collection '{collection_name}' in {end_time - start_time} seconds.")
        collection.flush()
        logger.info(f"Number of entities in the collection: {collection.num_entities}")
        
    except Exception as e:
        logger.error(f"Error inserting into Milvus: {str(e)}")
        raise

def is_low_value_article(page_content, templates=None):
    # Check the length of the article content
    content_length = len(page_content.strip())
    logger.debug(f"Content length: {content_length}")

    # Define thresholds for low-value articles
    if content_length < 250:  # Adjust the threshold as needed
        logger.info(f"Skipping low-value article with content length {content_length}")
        return True  # Skip article based on content length

    # Use templates to determine if the article should be skipped
    if templates:
        _, article_quality, skip_article = extract_whitelisted_templates(page_content)
        if skip_article:
            logger.info(f"Skipping article based on templates: {templates}")
            return True

    return False

#count out the article id
def load_article_counter(file_path=r"D:\scripts\article_counter.txt"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return int(f.read().strip())
    return 0

def save_article_counter(counter, file_path=r"D:\scripts\article_counter.txt"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
    with open(file_path, "w") as f:
        f.write(str(counter))

# Generate article_id
article_counter = load_article_counter()

def generate_article_id():
    global article_counter
    article_counter += 1
    save_article_counter(article_counter)
    return int(article_counter)  # Explicitly return as int

# Initialize article-specific chunk counter
chunk_counter = 0

def generate_chunk_id(article_id):
    global chunk_counter
    chunk_counter += 1
    return f"{article_id}:{chunk_counter}"

# Reset chunk counter when processing a new article
def reset_chunk_counter():
    global chunk_counter
    chunk_counter = 0


# Main program with selective template handling
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Wikipedia XML dump and insert into Milvus.")
    parser.add_argument("file_path", type=str, help="Path to the Wikipedia XML dump file.")
    parser.add_argument("collection_name", type=str, help="Name of the Milvus collection to insert data into.")
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.file_path):
        logger.error(f"File not found: {args.file_path}")
        exit(1)
    
    # Log the input file path and collection name
    logger.info(f"Processing file: {args.file_path}")
    logger.info(f"Inserting data into collection: {args.collection_name}")
    
    # Initialize constants
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_limit = 512
    
    # Stream and process XML data
    try:
        for idx, (title, article_text, sha1, edit_date, templates, article_quality) in enumerate(stream_xml(args.file_path)):
            # Skip redirect pages and other invalid articles
            if not article_text:
                logger.info(f"Skipping empty article: {title}")
                continue
                
            if is_low_value_article(article_text, templates):
                logger.info(f"Skipping low value article: {title}")
                continue
            
            logger.info(f"Processing Article {idx + 1}: {title}")
            
            # Generate a unique article_id only for articles we're actually processing
            article_id = generate_article_id()
            logger.debug(f"Generated article_id: {article_id}")
    
            # Reset the chunk counter for the new article
            reset_chunk_counter()

            # Generate the Kiwix URL for the article
            kiwix_url = construct_kiwix_url(title)
            
            # Clean the article text for embedding
            cleaned_content = clean_wikipedia_text(article_text)

            # Discard content after "See also" or similar sections
            cleaned_content = discard_post_see_also(cleaned_content)

            # Extract sections and process chunks
            section_data = extract_sections(cleaned_content)
            logger.info(f"Total sections extracted: {len(section_data)}")
            logger.debug(f"Extracted section titles: {[section['section_hierarchy'] for section in section_data]}")
            
            # Skip processing if no sections were extracted
            if not section_data:
                logger.warning(f"No sections extracted from article '{title}', skipping")
                continue
                
            chunks = chunk_article_with_metadata(title, section_data, token_limit, tokenizer, kiwix_url, sha1, article_quality, article_id)
            
            # Skip if no chunks were created
            if not chunks:
                logger.warning(f"No chunks created for article '{title}', skipping")
                continue
                
            logger.info(f"Total Chunks for '{title}': {len(chunks)}")

            # Embed chunks and insert into Milvus
            embeddings = embed_chunks(chunks, model_name)
            logger.info(f"Total Chunks Embedded for '{title}': {len(embeddings)}")
            
            insert_into_milvus(chunks, embeddings, collection_name=args.collection_name, edit_date=edit_date)
    except Exception as e:
        logger.error(f"Error processing articles: {e}")
        raise