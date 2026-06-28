import json
import sys
import re

def process_file(filepath, rules_path):
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
        
    if filepath.endswith('.ipynb'):
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        for cell in nb.get('cells', []):
            if 'source' in cell:
                new_source = []
                for line in cell['source']:
                    new_line = line
                    for rule in rules:
                        if rule.get('type') == 'regex':
                            new_line = re.sub(rule['pattern'], rule['replacement'], new_line)
                        else:
                            new_line = new_line.replace(rule['pattern'], rule['replacement'])
                    new_source.append(new_line)
                cell['source'] = new_source
                
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for rule in rules:
            if rule.get('type') == 'regex':
                content = re.sub(rule['pattern'], rule['replacement'], content)
            else:
                content = content.replace(rule['pattern'], rule['replacement'])
                
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

if __name__ == '__main__':
    process_file(sys.argv[1], sys.argv[2])
