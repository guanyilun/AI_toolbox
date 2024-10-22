"""
retrieve pdf bibcodes from the ADS library
"""
import os, os.path as op
import requests
import logging

logging.basicConfig(level=logging.INFO)

def download_pdf(bibcode, pdf_dir):
    """download pdf based on bibcode and save it to pdf_dir. To name
    the pdf file, use the bibcode as the filename with dot replaced by
    underscore. Return the filename of the saved pdf on success, or None
    """
    filename = bibcode.replace('.', '_') + '.pdf'
    filepath = op.join(pdf_dir, filename)
    if op.exists(filepath):
        logging.warning(f"PDF already exists for bibcode: {bibcode}")
        return filepath

    url = f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/EPRINT_PDF"
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filepath
    else:
        print(f"Failed to download PDF for bibcode: {bibcode}")
        return None

if __name__ == '__main__':
    import json
    import os, os.path as op
    import time

    class args:
        filename = 'spectral_distortion.json'
        pdf_dir = './pdfs/spectral_distortion'

    if not op.exists(args.pdf_dir):
        os.makedirs(args.pdf_dir)

    with open(args.filename, 'r') as f:
        paper_data = json.load(f)
    papers = paper_data['papers']

    # download pdfs
    for (i, bibcode) in enumerate(papers):
        try:
            filename = download_pdf(bibcode, args.pdf_dir)
        except Exception as e:
            print(f"Error downloading PDF for bibcode: {bibcode}")
            print(e)
            filename = None
        if filename is not None:
            papers[bibcode]['pdf'] = filename
            logging.info(f"{i+1:3d}/{len(papers)}: {bibcode} -> {filename}")
        time.sleep(5)
    with open(args.filename, 'w') as f:
        json.dump(paper_data, f, indent=2)