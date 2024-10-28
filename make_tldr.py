import json
import logging
import re
import time

logging.basicConfig(level=logging.INFO)

def make_tldr(lm, papers):
    prompt = f"""
    You are a helpful research assistant that provides TLDR summaries for papers given
    their titles, and abstracts. The papers will be provided in the following format:
    
    <paper><bibcode>bibcode</bibcode><title>title</title><abstract>abstract</abstract></paper>

    You will receive multiple papers in the same format. Please provide a TLDR summary
    for each paper in the following format:
    <paper><bibcode>bibcode</bibcode><tldr>tldr</tldr></paper>

    Here are the list of papers:
    """ 
    prompt += '\n'.join(
        [f"<paper><bibcode>{paper['bibcode']}</bibcode><title>{paper['title']}</title><abstract>{paper['abstract']}</abstract></paper>" for paper in papers]
    )
    prompt += "\nProvide your TLDR summaries:\n"
    res = lm.get_response(prompt)

    # extract tldr summaries
    rset = re.findall(r'<paper>(.*?)</paper>', res)
    tldrs = {}
    for r in rset:
        bibcode = re.search(r'<bibcode>(.*?)</bibcode>', r).group(1)
        tldr = re.search(r'<tldr>(.*?)</tldr>', r).group(1)
        tldrs[bibcode] = tldr
    
    return tldrs

if __name__ == '__main__':
    from literature_search import LM
    import dotenv
    dotenv.load_dotenv()

    class args:
        # filename = 'spectral_distortion.json'
        filename = 'birefringence.json'
        batch_size = 5
        sleep_time = 1

    with open(args.filename, 'r') as f:
        paper_data = json.load(f)
        
    papers = paper_data['papers']

    # model class: S > A > B > C > ...
    lm = {
        'S': LM("openai/gpt-4o"),
        'A': LM("openai/gpt-4o-mini"),
        'B': LM("nousresearch/hermes-3-llama-3.1-405b:free"), # 8k context, 4k output
        'C': LM("liquid/lfm-40b"),                            # 32k context, 32k output
    }

    bibcodes = list(papers.keys())
    bibcodes = [bibcode for bibcode in bibcodes if 'tldr' not in papers[bibcode]]
    batches = [bibcodes[i:i+args.batch_size] for i in range(0, len(bibcodes), args.batch_size)]

    tldrs = {}
    completed = []
    for batch in batches:
        logging.info(f"Processing batch of {len(batch)} papers...")
        batch = [papers[bibcode] for bibcode in batch]
        _tldr_batch = make_tldr(lm['A'], batch)
        for bibcode, tldr in _tldr_batch.items():
            papers[bibcode]['tldr'] = tldr
            logging.info(f"\t{bibcode}")
        with open(args.filename, 'w') as f:
            json.dump(paper_data, f, indent=2, sort_keys=True)
        completed += batch
        logging.info(f"\tProgress: {len(completed):3d}/{len(bibcodes)}")
        if args.sleep_time > 0: time.sleep(args.sleep_time)
