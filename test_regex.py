import re
cleaned = 'ENG 244-9 48.2/50 Run Rate 5.05 Topley24) C Overton 9 (17) P3 IND Bumrah 1-47 (9.2)'

over_match = re.search(r'\b(\d{1,2}(?:\.\d)?)\s*\/\s*50\b', cleaned)
print('Overs:', over_match.group(1) if over_match else None)

batsmen = re.findall(r'((?:[A-Z]\s+)?[A-Z][a-z]+)\s*(\d+)\s*[\(\[]?\s*(\d+)\s*[\)\]]', cleaned)
print('Batsmen:', batsmen)

# Testing bowler to make sure it doesn't match 'Bumrah 1-47'
bowler = re.findall(r'([A-Z][a-z]+)\s+(\d+)\-(\d+)\s*[\(\[]?\s*(\d+(?:\.\d)?)\s*[\)\]]', cleaned)
print('Bowler:', bowler)
