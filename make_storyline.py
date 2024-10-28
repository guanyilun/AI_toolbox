import json
import logging
import re
import time

logging.basicConfig(level=logging.INFO)

def make_storyline(lm, topic, papers):
    prompt = f"""
    You are a helpful research assistant that go through summaries of papers to retrieve
    relevant papers for a specific research topic. The papers will be provided in the following format:
    
    <paper><bibcode>bibcode</bibcode><title>title</title><tldr>tldr</tldr></paper>

    You will receive multiple papers in the same format. Please only keep the papers that are directly
    contributing to our knowedlge of the research topic. Please provide a short explanation for why you 
    think the paper is important contribution to the research topic. Please be very selective and focus
    on papers that matches the said topic. Please provide your response in the following format
    
    <paper><bibcode>bibcode</bibcode><reason>briefly describe the reason to keep this paper</reason></paper>
    
    If none of the papers are relevant, return an empty response.

    Here is the research topic to filter the papers:
    {topic}

    Here are the list of paper summaries:
    """ 
    papers = [paper for paper in papers if 'tldr' in paper]
    prompt += '\n'.join(
        [f"<paper><bibcode>{paper['bibcode']}</bibcode><title>{paper['title']}</title><tldr>{paper['tldr']}</tldr></paper>" for paper in papers]
    )
    prompt += "\nProvide your response below:\n"
    res = lm.get_response(prompt)

    # extract tldr summaries
    rset = re.findall(r'<paper>(.*?)</paper>', res)
    storyline = {}
    for r in rset:
        bibcode = re.search(r'<bibcode>(.*?)</bibcode>', r).group(1)
        reason = re.search(r'<reason>(.*?)</reason>', r).group(1)
        storyline[bibcode] = reason
    return storyline

if __name__ == '__main__':
    from literature_search import LM
    import dotenv
    dotenv.load_dotenv()

    class args:
        filename = 'birefringence.json'
        batch_size = 20 
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
    bibcodes = [bibcode for bibcode in bibcodes if 'tldr' in papers[bibcode]]
    batches = [bibcodes[i:i+args.batch_size] for i in range(0, len(bibcodes), args.batch_size)]

    topics = [
        # 'using cosmic birefringence to study axion-like particles'
        # 'observational constraint on cosmic birefringence using CMB',
        'impact from galactic foreground on cosmic birefringence measurement when measuring isotropic birefringence using CMB polarization',
    ]

    if 'storylines' not in paper_data:
        storylines = {t: {} for t in topics}
    else:
        storylines = paper_data['storylines']
    # force re-run
    # storylines = {t: {} for t in topics}
    completed = []
    for t in topics:
        if t not in storylines: storylines[t] = {}
        logging.info(f"Building storyline around topic: {t}")
        for batch in batches:
            logging.info(f"Processing batch of {len(batch)} papers...")
            batch = [papers[bibcode] for bibcode in batch if bibcode not in storylines[t]]
            if len(batch) == 0: continue
            _storyline = make_storyline(lm['S'], t, batch)
            if len(_storyline) == 0: 
                logging.info("\tNo relevant papers found.")
                continue
            for bibcode, reason in _storyline.items():
                if bibcode not in papers: continue
                storylines[t][bibcode] = { 'reason': reason, 'tldr': papers[bibcode].get('tldr', '')}
                logging.info(f"\t{bibcode}: {reason}")
            
            # save progress
            paper_data['storylines'] = storylines
            with open(args.filename, 'w') as f:
                json.dump(paper_data, f, indent=2, sort_keys=True)
            completed += batch
            logging.info(f"\tProgress: {len(completed):3d}/{len(bibcodes)}")
            if args.sleep_time > 0: time.sleep(args.sleep_time)
