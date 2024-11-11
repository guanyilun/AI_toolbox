"""
Due to restriction of context size, a long-form reading such as a book
over hundreds pages cannot be effectively summarized in one go. One has
to devise a scheme to process the text in batches each tailored to fit within
the context and integrate the integration into another long-form response that
otherwise couldn't fit into the output length either. How to achieve this?
think of it as a data compression problem? One needs to create a hierachical
embedding of the data and selectively navigate the information tree.
"""

import json
import logging
import re
import time
import traceback
from typing import List, Optional
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)

def make_tldr(lm, text, directive):
    prompt = f"""
    You are a helpful assistant that go through batches of text to produce summary of text based on a given directive provided within
    <directive>...</directive> tags. The text will be provided within
    <text>...</text> tags. Note that the text may contain random garbage text that is not relevant to the directive. Please identify only paragraph that is relevant to the directive and produce summaries for each relevant non-garbage textual paragraph. Please provide the summaries in the following format:

    <p><summary>summary of relevant paragraph 1</summary><fulltext>full text (clean) of relevant paragraph 1</fulltext></p>
    <p><summary>summary of relevant paragraph 2</summary><fulltext>full text (clean) of relevant paragraph 2</fulltext></p>
    ...
    
    Here is the directive:
    <directive>{directive}</directive>
    
    Here is the text to summarize:
    <text>{text}</text>

    Your summaries:
    """ 
    res = lm.get_response(prompt)

    # extract tldr summaries
    rset = re.findall(r'<p>(.*?)</p>', res)

    out = []
    for r in rset:
        try:
            summary = re.search(r'<summary>(.*?)</summary>', r).group(1)
            fulltext = re.search(r'<fulltext>(.*?)</fulltext>', r).group(1)
            out.append({
                'summary': summary,
                'fulltext': fulltext,
                'directive': directive
            })
        except:
            logging.error(f"\tError parsing response")
    return out

def write_tldrs(tldrs, filename):
    slim = [ res['summary'] for res in tldrs ]
    filename_slim = filename.replace('.json', '_slim.json')
    with open(filename_slim, 'w') as f:
        json.dump(slim, f, indent=2)
    with open(filename, 'w') as f:
        json.dump(tldrs, f, indent=2)

from collections import namedtuple
from typing import List

# Since namedtuple is immutable, we need to create a custom class that extends it
# to handle the mutable subsections list
class Section(namedtuple('Section', ['section_id', 'title', 'subsections'])):
    def __new__(cls, section_id: str, title: str, subsections: List["Section"] = None):
        return super().__new__(cls, section_id, title, subsections or [])


def parse_sections(text: str) -> List[Section]:
    # Remove newlines and extra spaces to simplify parsing
    text = re.sub(r'\s+', ' ', text.strip())
    
    def parse_section(text: str, start: int = 0) -> tuple[Optional[Section], int]:
        # Match opening section tag with id and title attributes
        section_start = re.match(
            r'<section\s+id="([^"]+)"\s+title="([^"]+)">', 
            text[start:]
        )
        
        if not section_start:
            return None, start
            
        # Create new section
        current_section = Section(
            section_id=section_start.group(1),
            title=section_start.group(2)
        )
        
        # Move pointer past opening tag
        pos = start + section_start.end()
        
        # Parse subsections recursively until we hit closing tag
        while pos < len(text):
            if text.startswith('</section>', pos):
                return current_section, pos + 10
                
            # Try to parse subsection
            subsection, new_pos = parse_section(text, pos)
            if subsection:
                current_section.subsections.append(subsection)
                pos = new_pos
            else:
                pos += 1
                
        return current_section, pos

    # Parse all top-level sections
    sections = []
    pos = 0
    while pos < len(text):
        section, new_pos = parse_section(text, pos)
        if section:
            sections.append(section)
            pos = new_pos
        else:
            pos += 1
            
    return sections

def make_outline(lm, tldrs, topic):
    prompt = f"""
    You are a helpful research assistant that go through batches of text summaries to produce an outline of the text
    that can be used as a basis to build a comprehensive article about a given topic. The topic is provided within
    <topic>...</topic> tags. The text summaries will be provided in the following format:

    <summary>summary 1</summary>
    <summary>summary 2</summary>

    Please provide an outline of the text that can be used to write a full article about the given topic.
    The outline should be in the following hierarchical format:

    <section id="id of the section" title="title of the section">
        <section id=..., title=...>...</section>
        <section id=..., title=...>...</section>
    </section>
    <section id="id of the section" title="title of the section">
        <section id=..., title=...>...</section>
        <section id=..., title=...>...</section>
    </section>
    ...

    The id should be unique for each section using lowercase letters connected by hyphens. The title should be a brief
    description of the content of the section.

    Here is the topic:
    <topic>{topic}</topic>
    
    Here are the text summaries:
    """ 
    prompt += '\n'.join(
        [f"<summary>{tldr['summary']}</summary>" for tldr in tldrs]
    )
    prompt += "\nProvide your outline below:\n"
    
    res = lm.get_response(prompt)

    # parse outline
    try:
        outline = parse_sections(res)
    except:
        logging.error(f"\tError parsing response: {res}")
        traceback.print_exc()
        outline = None

    out = {
        'topic': topic,
        'text': res,
        'outline': outline
    }

    return out

def merge_outlines(lm, outlines, topic):
    prompt = f"""
    In order to produce a coherent write-up of a given topic,
    you are tasked to build an comprehensive outline of an article
    that can be used to produce a comprehensive and coherent
    article on the given topic. You will be provided with multiple 
    outlines each generated from a subset of text. They can be
    overlaping or complementing. Please incorporate the given 
    outlines into one single coherent outline as the basis of the 
    article. The topic is provided within <topic>...</topic> tags.

    Each outline will be in the following format:
    <outline>
    
    <section id="id of the section" title="title of the section">
        <section id=..., title=...>...</section>
        <section id=..., title=...>...</section>
    </section>
    <section id="id of the section" title="title of the section">
        <section id=..., title=...>...</section>
        <section id=..., title=...>...</section>
    </section>
    ...

    </outline>

    where the id should be unique for each section using lowercase letters connected by hyphens. The title should be a brief
    description of the content of the section.

    The output outline should be in the same format.
    
    Here is the topic:
    <topic>{topic}</topic>
    
    Here are the outlines:
    """
    prompt += '\n'.join(
        [f"<outline>{outline['text']}</outline>" for outline in outlines]
    )
    prompt += "\nProvide your best outline below:\n"
    
    res = lm.get_response(prompt)

    # parse outline
    try:
        outline = parse_sections(res)
    except:
        logging.error(f"\tError parsing response: {res}")
        traceback.print_exc()
        outline = None

    out = {
        'topic': topic,
        'text': res,
        'outline': outline
    }
    return out


if __name__ == '__main__':
    from literature_search import LM
    import dotenv
    import os
    dotenv.load_dotenv()

    class args:
        filename = 'FDR.md'
        batch_size = 200  # lines per batch
        output = 'FDR_tldrs.json'
        # step = 'outline'
        step = 'merge_outlines'
        tldr_batch_size = 50
        outline_batch_size = 5

    # model class: S > A > B > C > ...
    lm = {
        'S': LM("openai/gpt-4o"),
        'A': LM("openai/gpt-4o-mini"),
        'B': LM("liquid/lfm-40b:free"),                            # 32k context, 4k output
    }

    if args.step == 'tldr':
        with open(args.filename, 'r') as f:
            lines = f.readlines()

        batches = [lines[i:i+args.batch_size] for i in range(0, len(lines), args.batch_size)]

        directive = "the personal character of FDR and how it influenced his presidency. What critisms has he faced and how did he respond to them?"

        if os.path.exists(args.output):
            with open(args.output, 'r') as f:
                tldrs = json.load(f)
        else:
            tldrs = []
        for i, batch in enumerate(batches[50:]):
            prompt = ''.join(batch)
            res = make_tldr(lm['A'], prompt, directive)
            res = [ {**r, 'batch': i} for r in res ]
            logging.info(f"Processing batch {i+1}/{len(batches)}: {len(prompt)} characters, {len(res)} summaries")
            if len(res) == 0:
                logging.debug(f"Skipping batch {i+1}/{len(batches)}: no summaries found")
                continue
            tldrs.extend(res)
            write_tldrs(tldrs, args.output)
            time.sleep(2)

    elif args.step == 'outline':
        # with a large set of paragraph summaries, in order to produce a 
        # coherent summary of the full text, one needs to generate an 
        # outline of ideas and then fill in the details of each section
        if os.path.exists(args.output):
            with open(args.output, 'r') as f:
                tldrs = json.load(f)
        else:
            logging.error(f"Cannot find tldrs file: {args.output}")
            exit(1)
        
        # split into batches to process
        tldrs_batches = [tldrs[i:i+args.tldr_batch_size] for i in range(0, len(tldrs), args.tldr_batch_size)]
        outlines = []
        for i, tldrs in enumerate(tldrs_batches):
            logging.info(f"Processing batch {i+1}/{len(tldrs_batches)}")
            outline = make_outline(lm['A'], tldrs, "FDR's character and presidency")
            if isinstance(outline, list):
                outlines.extend(outline)
            else:
                outlines.append(outline)

            # save progress
            with open(args.output.replace('.json', '_outline.json'), 'w') as f:
                json.dump(outlines, f, indent=2)

            time.sleep(2)

    elif args.step == 'merge_outlines':
        ifile = args.output.replace('.json', '_outline.json')
        if not os.path.exists(ifile):
            logging.error(f"Cannot find outline file: {ifile}")
            exit(1)
        with open(ifile, 'r') as f:
            outlines = json.load(f)

        best_outline = merge_outlines(lm['S'], outlines, "FDR's character and presidency")

        with open(args.output.replace('.json', '_merged_outline.json'), 'w') as f:
            json.dump(best_outline, f, indent=2)

        # todo:
        # - map each tldr to a section in the outline
        # - go through each section and fill in the details
        # - save the final document as a markdown file
    