#!/usr/bin/env python

"""Add an event to Apple Calendar."""

import dotenv

dotenv.load_dotenv()
import re

# from openai3p import OpenAI3P # 3rd party OpenAI config
from openai import OpenAI
from datetime import datetime


def get_response(
    client,
    prompt,
    system_message=None,
    max_tokens=5000,
    model="openai/gpt-4o-mini",  # , model="glm-4-flashx" # 3rd party OpenAI config
):
    if system_message is None:
        system_message = (
            "You are a helpful assistant that answer questions and provide guidance."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def make_applescript(client, event_info):
    script = r"""
    set eventTitle to "Talk by Bob Frey"
    set eventDate to current date
    set year of eventDate to 2024
    set month of eventDate to October
    set day of eventDate to 16
    set time of eventDate to (15 * hours) -- 15:00

    set eventEndDate to current date
    set year of eventEndDate to 2024
    set month of eventEndDate to October
    set day of eventEndDate to 16
    set time of eventEndDate to (16 * hours) -- 16:00

    set eventLocation to "BA 5187"
    set eventDescription to "A talk by Bob Frey (Cornell University) followed by a reception with light refreshments."

    tell application "Calendar"
        tell calendar "Work"
            set newEvent to make new event with properties {summary:eventTitle, start date:eventDate, end date:eventEndDate, location:eventLocation, description:eventDescription}
        end tell
    end tell
    """
    prompt = f"""
    Here is a brief description of an event that I want to attend. 
   
    Event info: {event_info} 
    
    Please help me build an AppleScript that adds the event to my Apple Calendar called Work. 

    Here is an example applescript that you can derive
    from:

    {script}
    
    Please note the current year is {datetime.now().year}.

    Please provide the updated AppleScript code in markdown format wrapped
    with ```applescript at the beginning and ``` at the end.


    Modified AppleScript:
    """
    response = get_response(client, prompt)
    return response


def parse_applescript(script):
    """parse script from markdown format"""
    # Regular expression pattern to match markdown between triple backticks
    pattern = r"```(?:applescript)?\s*([\s\S]*?)\s*```"

    # Find all matches in the response
    matches = re.findall(pattern, script, re.IGNORECASE)

    if matches:
        return matches[0].strip()
    else:
        return None


if __name__ == "__main__":
    import tempfile
    import os
    import subprocess

    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    print("Please provide a brief description of the event.")
    print("Enter your description, and type 'END' on a new line when you're finished:")
    
    event_details = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        event_details.append(line)
    
    event_details = "\n".join(event_details)
    script = make_applescript(client, event_details)
    script = parse_applescript(script)
    if script is None:
        raise ValueError("Failed to parse AppleScript")

    print("AppleScript:\n", script)

    proceed = input("Do you want to execute the AppleScript? (y/n): ")
    if proceed.lower() != "y":
        print("Exiting...")
        exit(0)

    # save the AppleScript to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".scpt", dir="/tmp", delete=False
    ) as temp_file:
        temp_file.write(script)
        temp_file_path = temp_file.name

    print(f"AppleScript saved to: {temp_file_path}")
    # execute the saved AppleScript
    try:
        result = subprocess.run(
            ["osascript", temp_file_path], capture_output=True, text=True, check=True
        )
        print("AppleScript executed successfully.")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error executing AppleScript:")
        print("Error message:", e.stderr)
    finally:
        # clean up the temporary file
        os.remove(temp_file_path)
        print(f"Temporary file removed: {temp_file_path}")
