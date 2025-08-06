#!/usr/bin/env python3
import sys
import re
import json

def fix_json_quotes(input_file, output_file=None):
    """
    Fix JSON file by replacing single quotes with double quotes
    
    Args:
        input_file: Path to the input JSON file with single quotes
        output_file: Path to the output JSON file with double quotes (optional)
                     If not provided, will overwrite the input file
    """
    if output_file is None:
        output_file = input_file + ".fixed"
        
    print(f"Reading from {input_file}...")
    
    try:
        with open(input_file, 'r') as f:
            content = f.read()
        
        # Check if the content starts with a valid JSON character
        if not content.strip().startswith('{') and not content.strip().startswith('['):
            print("Adding missing JSON structure...")
            content = '{' + content
        
        if not content.strip().endswith('}') and not content.strip().endswith(']'):
            print("Adding missing closing brace...")
            content = content + '}'
        
        # Replace single quotes with double quotes
        print("Replacing single quotes with double quotes...")
        fixed_content = content.replace("'", '"')
        
        # Fix any potential issues with double quotes
        fixed_content = fixed_content.replace('""', '"')
        
        # Try to parse the fixed JSON to make sure it's valid
        try:
            json_data = json.loads(fixed_content)
            print("Successfully fixed JSON format.")
            
            # Write the fixed JSON to the output file
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=2)
                
            print(f"Fixed JSON saved to {output_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error: Could not fix JSON format. {str(e)}")
            
            # Try a more sophisticated approach for fixing quotes in keys and values
            print("Trying regex-based approach...")
            
            # Fix keys
            regex_fixed = re.sub(r"'([^']*)':", r'"\1":', content)
            # Fix values
            regex_fixed = re.sub(r": '([^']*)'", r': "\1"', regex_fixed)
            # Fix values in lists
            regex_fixed = re.sub(r", '([^']*)'", r', "\1"', regex_fixed)
            regex_fixed = re.sub(r"\['([^']*)'\]", r'["\1"]', regex_fixed)
            
            try:
                json_data = json.loads(regex_fixed)
                print("Successfully fixed JSON format using regex approach.")
                
                # Write the fixed JSON to the output file
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
                print(f"Fixed JSON saved to {output_file}")
                return True
                
            except json.JSONDecodeError as e:
                print(f"Error: Could not fix JSON format using regex approach. {str(e)}")
                
                # Last resort: manual character-by-character replacement
                print("Trying manual character replacement...")
                
                # Write the best attempt to a file for manual inspection
                with open(output_file + ".partial", 'w') as f:
                    f.write(regex_fixed)
                
                print(f"Best attempt saved to {output_file}.partial")
                print("You may need to manually fix the JSON file.")
                return False
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_json_quotes.py <input_file> [output_file]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = fix_json_quotes(input_file, output_file)
    sys.exit(0 if success else 1)
