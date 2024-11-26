import re

SELF_DIAGNOSIS_PROMPT_TEMPLATE = '''
Fact-checking involves accessing the veracity of a given text, which could be a statement, claim, paragraph, or a generated response from a large language model.

For more complex fact-checking tasks, the input text is often broken down into a set of manageable, verifiable and self-contained sub-claim, a process called 'Input Decomposition'. Each sub-claim is required to be *self-contained*, meaning it is completely interpretable without the original text and other sub-claims. Each sub-claim is verified independently, and the results are combined to assess the overall veracity of the original input. The decomposition error categories are defined as below:

"""
# Error Categories in Decomposition

### Omission of Context Information

Failure to include critical elements necessary for accurate understanding or verification of the claim. This category encompasses:

#### Missing Core Claims or Key Details

- **Definition**: Exclusion of background or situational details that provide necessary understanding of the claim.

#### Missing Logical Relationships**

- **Definition**: Failure to include cause-and-effect links (Causal Relationship), comparisons(Comparative Relationship), or contrasts that explain how different parts of the claim relate to each other. Key relationships information lost.


### Ambiguity

The decomposition is unclear or vague, leading to multiple interpretations and making accurate understanding or verification difficult. This includes:

#### Vague Language

- **Definition**: Use of terms or references that do not convey specific meaning, making the claim unclear.
For examples:
    - Pronouns without Clear Antecedents (e.g., “his”, “they”, “her”)
    - Unknown entities (e.g., “this event”, “the research”, “the invention”)
    - Non-full names (e.g., “Jeff...” or “Bezos...” when referring to Jeff Bezos)


#### Multiple Interpretations
- **Definition**: Phrases that can be understood in different ways due to lack of specificity or precision.

### Over-decomposition

Including unnecessary repetition or excessively breaking down the claim, leading to complexity or alterating the original meaning. This category covers:

---

#### Redundant Information

- **Definition**: Repetition of the same information without adding value. Sub-claims that cover the same content areas without providing distinct insights, leading to repetition. Unverifiable information also add no value and should be considered as redundant.


#### Over-Decomposition

- **Definition**: Fragmenting the claim into too many sub-claims, causing loss or alteration of original meaning.


### Alteration of original meaning
- **Definition**: The decomposition changes the original meaning of the claim by introducing excessive, hallucinated, fabricated, or contradictory information that is not present in the original input.
"""





Your task is to evaluate whether the following input decomposition is 'Good' or 'Problematic' or 'No need for decomposition'.

Please give your final judgment and support it with your justification. If the decomposition is problematic, identify the type of error involved and suggest the set of refined decomposition when necessary. If the given text is already concise and clear enough and does not need further decomposition, you can use the given claim as the `Refined Decomposition` and mark the Judgement as 'No need for decomposition'.

Use tripple backticks to enclose the reasoning process, error type, judgment and refined decomposition. Your response MUST follow this format:

### Reasoning
```
[Provide a step-by-step explanation of your reasoning.]
```

### Error Type
```
[Identify the specific type of error, if any, in the decomposition.]
```

### Judgment
```
[Conclude whether the decomposition is 'Good' or 'Problematic' or 'No need for decomposition'.]
```

### Refined Decomposition
```
[Provide the refined set of sub-claims.]
```


Here are some examples:


### Given text
```
Images were required to be censored as part of a group exhibition at The Sydney Fringe.
```

### Decomposition
```
Images were required to be censored as part of a group exhibition at The Sydney Fringe.
```

### Reasoning
```
The given text states that images were required to be censored as part of a group exhibition at The Sydney Fringe. To create an effective decomposition, it is important to maintain the key elements—what was required (censorship of images), the context (as part of a group exhibition), and the location/event (The Sydney Fringe).

The decomposition should ensure that each sub-claim is self-contained and avoids over-decomposition or omission of necessary details. In this case, it is sufficient to have one sub-claim that maintains the relationship between the images, censorship requirement, and the event.
```

### Judgement
```
Good
```

### Refined Decomposition
```
- Images were required to be censored as part of a group exhibition at The Sydney Fringe.
```

==========

### Given text
```
Due to Imran Khan's criticism of Macron's comments on Islam, French authorities cancelled the visas of 183 Pakistani citizens and deported 118 from the country.
```

### Decomposition
```
- French authorities cancelled the visas of 183 Pakistani citizens.
- French authorities deported 118 Pakistani citizens from the country.
```

### Reasoning
```
The decomposition breaks the given text into two sub-claims about the actions of the French authorities. However, it fails to incorporate the causal relationship that links Imran Khan's criticism of Macron's comments on Islam to the actions of visa cancellation and deportation. This omission makes the decomposition incomplete, as the causal context is crucial for understanding why these actions were taken.
```

### Error Type
```
Omission of Context Information - Missing Logical Relationships
```

### Judgment
```
Problematic
```

### Refined Decomposition
```
- Due to Imran Khan's criticism of Macron's comments on Islam, French authorities cancelled the visas of 183 Pakistani citizens.
- Due to Imran Khan's criticism of Macron's comments on Islam, French authorities deported 118 Pakistani citizens from the country.
```

==========

### Given text
```
President Ferdinand Marcos and Dr. Jose Rizal established and founded the WORLD BANK and International Monetary Fund.
```

### Decomposition
```
- President Ferdinand Marcos did not establish or found the WORLD BANK.
- Dr. Jose Rizal did not establish or found the WORLD BANK.
- President Ferdinand Marcos did not establish or found the International Monetary Fund.
- Dr. Jose Rizal did not establish or found the International Monetary Fund.
```

### Reasoning
```
The original text claims that President Ferdinand Marcos and Dr. Jose Rizal established and founded the World Bank and the International Monetary Fund. The decomposition provided, however, negates these claims by stating that they did not establish or found these institutions. This reversal changes the original meaning of the text. So the error here is *alteration of original meaning**.
```

### Error type
```
Alteration of Original Meaning
```

### Judgement
```
Problematic
```

### Improved Decomposition
```
- President Ferdinand Marcos and Dr. Jose Rizal established and founded the World Bank.
- President Ferdinand Marcos and Dr. Jose Rizal established and founded the International Monetary Fund.
```

==========

### Given text
```
The smallest ocean in the world is the Arctic Ocean. It is located in the northernmost part of the Earth and is surrounded by the land masses of North America, Europe, and Asia. The Arctic Ocean covers an area of about 14.05 million square kilometers.
```

### Decomposition
```
- The smallest ocean in the world is the Arctic Ocean.
- The Arctic Ocean is surrounded by the land masses of North America.
- The Arctic Ocean is surrounded by the land masses of Europe.
- The Arctic Ocean is surrounded by the land masses of Asia.
- The Arctic Ocean covers an area of about 14.05 million square kilometers.
```

### Reasoning
```
The information about the Arctic Ocean being surrounded by the land masses of North America, Europe, and Asia is broken down into three separate sub-claims. This is unnecessary and results in redundancy, as these details could be combined into a single sub-claim without losing clarity or meaning. Decomposing into separate sub-claims could leads to mis-interpretation as 'is surrounded by' should refer to multiple continents in this context. So the first two issues are **redundant information** and **alteration of original meaning**. The decomposition also omits the context that state the Arctic Ocean is located in the northernmost part of the Earth, which need to be verified. So the third issue is **missing core claims or key details**.
```

### Error Type
```
- Over-Decomposition: Redundant Information
- Alteration of Original Meaning
- Omission of Context Information: Missing Core Claims or Key Details
```

### Judgment
```
Problematic
```

### Refined Decomposition
```
- The smallest ocean in the world is the Arctic Ocean.
- The Arctic Ocean is located in the northernmost part of the Earth.
- The Arctic Ocean is surrounded by the land masses of North America, Europe, and Asia.
- The Arctic Ocean covers an area of about 14.05 million square kilometers.
```

==========

### Given text
```
As an AI language model, I cannot make subjective statements or predictions about the success or failure of a particular aircraft. However, it is worth noting that the Sukhoi Su-57 has faced some challenges in its development, including delays and budget constraints. Additionally, there are concerns about the aircraft's engine and stealth capabilities. Ultimately, the success of the Su-57 will depend on a variety of factors, including its performance in combat situations and its ability to meet the needs of the Russian military.
```

### Extracted claims
```
- I am an AI language model.
- I cannot make subjective statements.
- I cannot make predictions about the success of a particular aircraft.
- I cannot make predictions about the failure of a particular aircraft.
- The Sukhoi Su-57 has faced challenges in its development.
- The challenges faced by the Sukhoi Su-57 include delays.
- The challenges faced by the Sukhoi Su-57 include budget constraints.
- There are concerns about the aircraft's engine.
- There are concerns about the aircraft's stealth capabilities.
- The success of the Su-57 will depend on a variety of factors.
- One factor is its performance in combat situations.
- Another factor is its ability to meet the needs of the Russian military.
```

### Reasoning
```
The focus of the decomposition should be on the verifiable and informative content of the text, while excluding un-verifiable claims such as "I am an AI language model" and other non-informative statements. The original claim contains verifiable information regarding the challenges faced by the Sukhoi Su-57, concerns about its capabilities, and factors that will determine its success. Decomposing it into too many sub-claims, such as separately listing delays, budget constraints, engine concerns, and stealth capabilities, may lead to **over-decomposition**. So the issue here is **redundant information**. Since these details are interrelated and flow naturally as part of a coherent explanation, they can be grouped more effectively for simplicity and clarity.
```

### Error Type
```
- Over-Decomposition: Redundant Information
```

### Judgment
```
Problematic
```

### Refined Decomposition
```
- The Sukhoi Su-57 has faced challenges in its development, including delays and budget constraints.
- There are concerns about the Sukhoi Su-57's engine and stealth capabilities.
- The success of the Sukhoi Su-57 will depend on its performance in combat situations and its ability to meet the needs of the Russian military.
```

==========

### Given text
```
In a letter to Steve Jobs, Sean Connery refused to appear in an apple commercial.
```

### Decomposition
```
- Sean Connery refused to appear in an Apple commercial in a letter to Steve Jobs.
```

### Reasoning
```
As the original text is clear and concise enough, the provided decomposition maintains all the key details: Sean Connery's refusal, the communication method (a letter), and the context of the Apple commercial. The decomposition is self-contained and does not omit any necessary details or over-explain. Therefore, no further decomposition is needed.
```

### Error Type
```
None
```

### Judgment
```
Good
```

### Refined Decomposition
```
In a letter to Steve Jobs, Sean Connery refused to appear in an apple commercial.
```

==========

### Given text
```
President Muhammadu Buhari sign a new Police bill to allow officers arrest without a warrant
```

### Decomposition
```
- President Muhammadu Buhari exists.
- President Muhammadu Buhari is the head of state of Nigeria.
- A new bill related to the police was signed by President Muhammadu Buhari.
- The new police bill allows police officers to make arrests.
- The new police bill removes the requirement for a warrant before making arrests.
- The new police bill was signed by President Muhammadu Buhari on a date not specified in the text.
```

### Reasoning
```
The given decomposition includes excessive breakdown of information that could be combined for better clarity. For instance, stating that "President Muhammadu Buhari exists" and "President Muhammadu Buhari is the head of state of Nigeria" are irrelevant for verifying the core claim and add unnecessary complexity, leading to **redundant information**.
```

### Error Type
```
- Over-decomposition: Redundant Information 
```

### Judgement
Problematic

### Refined Decomposition
```
President Muhammadu Buhari signed a new police bill that allows officers to arrest without a warrant.
```

==========

### Given text
```
Sobral argued at his winners' press conference, however, that it held a humanitarian message, not political one.
```

### Decomposition
```
- Sobral won an event.
- A press conference was held.
- Sobral spoke at the press conference.
- Sobral argued a point at the press conference.
- The point Sobral argued was not a political message.
- The point Sobral argued was a humanitarian message.
```

### Reasoning
```
The given decomposition excessively breaks down the information from the original text into multiple sub-claims, which results in over-decomposition and redundant information. The decomposition includes unnecessary details, such as "Sobral won an event" and "A press conference was held," which do not add any new verifiable information about the main claim. Additionally, the separation of "The point Sobral argued was not a political message" and "The point Sobral argued was a humanitarian message" is redundant, as these could be expressed in a more concise and coherent manner.

A more effective decomposition would focus on the key message of Sobral's argument about the nature of the message (humanitarian vs. political), while keeping the context of the winners' press conference intact.
```

### Error Type
```
- Over-Decomposition: Redundant Information
```

### Judgment
```
Problematic
```

### Refined Decomposition
```
- Sobral argued at his winners' press conference that the message was humanitarian, not political.
```

==========

### Given text
```
Other Countries Have Proved that the U.S. FDA Was Wrong to Withdraw Approval Of Hydroxychloroquine as COVID Drug.
```

### Decomposition
```
- Other countries have proved something.
- The U.S. FDA withdrew approval of hydroxychloroquine.
- Hydroxychloroquine was considered a COVID drug.
```

### Reasoning
```
The decomposition lacks sufficient detail to fully convey the original claim and results in ambiguity and omission of important logical relationships. "Other countries have proved something" is vague and does not communicate the specific point being argued. Additionally, the connection between the actions of other countries and the U.S. FDA's decision is lost in the decomposition, which is a key part of the claim.

The core claim involves other countries proving that the U.S. FDA made an incorrect decision about hydroxychloroquine, implying a contradiction or difference in judgment. This context should be retained to accurately reflect the claim.
```

### Error Type
```
- Omission of Context Information: Missing Core Claims or Key Details
- Ambiguity: Vague Language
```

### Judgment
```
Problematic
```

### Refined Decomposition
```
- The U.S. FDA withdrew approval of hydroxychloroquine as a COVID drug.
- Other countries have proved that the U.S. FDA's decision to withdraw approval of hydroxychloroquine was wrong.
```

==========

### Given text
```
{input_text}
```

### Decomposition
```
{sub_claims}
```
'''



DETECTION_PROMPT_TEMPLATE = '''
Fact-checking involves accessing the veracity of a given text, which could be a statement, claim, paragraph, or a generated response from a large language model.

For more complex fact-checking tasks, the input text is often broken down into a set of manageable, verifiable and self-contained sub-claim, a process called 'Input Decomposition'. Each sub-claim is required to be *self-contained*, meaning it is completely interpretable without the original text and other sub-claims. Each sub-claim is verified independently, and the results are combined to assess the overall veracity of the original input. The decomposition error categories are defined as below:


"""
# Error Categories in Decomposition

---

### Omission of Context Information

Failure to include critical elements necessary for accurate understanding or verification of the claim. This category encompasses:

#### Missing Core Claims or Key Details

- **Definition**: Exclusion of essential background or situational details that provide the necessary context for understanding the claim. Without these elements, the sub-claims become incomplete or misleading.

#### Missing Logical Relationships

- **Definition**: Omission of relationships such as cause-and-effect links (Causal Relationship), comparisons (Comparative Relationship), or contrasts that explain how different parts of the claim relate to each other. These relationships are vital for understanding the interactions within the original claim.

---

### Ambiguity

The decomposition is unclear or vague, leading to multiple interpretations and making accurate understanding or verification difficult. This includes:

#### Vague Language

- **Definition**: Use of terms or references that do not convey specific meaning, making the claim unclear. Examples include:
  - Pronouns without clear antecedents (e.g., "his," "they," "her").
  - Unspecified entities (e.g., "this event," "the research," "the invention").
  - Non-full names (e.g., "Jeff..." or "Bezos..." instead of "Jeff Bezos").

---

### Over-Decomposition

Unnecessarily breaking down the claim into too many sub-claims or repeating the same information without adding value, leading to increased complexity or misinterpretation of the original meaning. This category covers:

#### Redundant Information

- **Definition**: Repetition of information that does not provide additional value. This includes sub-claims that cover the same content without offering distinct insights or including unverifiable information that adds no value.

#### Excessive Fragmentation

- **Definition**: Fragmenting the claim into too many sub-claims, resulting in unnecessary complexity, which can lead to a loss or alteration of the original meaning. This often involves breaking down content that could be expressed concisely.

---

### Alteration of Original Meaning

- **Definition**: The decomposition introduces excessive, hallucinated, fabricated, or contradictory information that changes the original meaning of the claim. This includes misrepresentation of the original statement or adding elements that were not present in the initial input.

---

"""

Your task is to evaluate whether the following input decomposition is 'Acceptable' or 'Problematic'.

Please give your final judgment and support it with your justification. If the decomposition is problematic, identify the error(s) involved.

Use tripple backticks to enclose the reasoning process, error type, and judgment. Your response MUST follow this format:

### Reasoning
```
[Provide a step-by-step explanation of your reasoning.]
```

### Error Type
```
[Identify the specific type of error, if any, in the decomposition.]
```

### Judgment
```
[Conclude whether the decomposition is 'Acceptable' or 'Problematic'.]
```

Here are some examples:

### Given text
```
Due to Imran Khan's criticism of Macron's comments on Islam, French authorities cancelled the visas of 183 Pakistani citizens and deported 118 from the country.
```

### Decomposition
```
- French authorities cancelled the visas of 183 Pakistani citizens.
- French authorities deported 118 Pakistani citizens from the country.
```

### Reasoning
```
The decomposition breaks the given text into two sub-claims about the actions of the French authorities. However, it fails to incorporate the causal relationship that links Imran Khan's criticism of Macron's comments on Islam to the actions of visa cancellation and deportation. This omission makes the decomposition incomplete, as the causal context is crucial for understanding why these actions were taken.
```

### Error Type
```
Omission of Context Information - Missing Logical Relationships
```

### Judgment
```
Problematic
```

==========

### Given text
```
President Ferdinand Marcos and Dr. Jose Rizal established and founded the WORLD BANK and International Monetary Fund.
```

### Decomposition
```
- President Ferdinand Marcos did not establish or found the WORLD BANK.
- Dr. Jose Rizal did not establish or found the WORLD BANK.
- President Ferdinand Marcos did not establish or found the International Monetary Fund.
- Dr. Jose Rizal did not establish or found the International Monetary Fund.
```

### Reasoning
```
The original text claims that President Ferdinand Marcos and Dr. Jose Rizal established and founded the World Bank and the International Monetary Fund. The decomposition provided, however, negates these claims by stating that they did not establish or found these institutions. This reversal changes the original meaning of the text. So the error here is *alteration of original meaning**.
```

### Error type
```
Alteration of Original Meaning
```

### Judgement
```
Problematic
```

==========

### Given text
```
Images were required to be censored as part of a group exhibition at The Sydney Fringe.
```

### Decomposition
```
Images were required to be censored as part of a group exhibition at The Sydney Fringe.
```

### Reasoning
```
The given text states that images were required to be censored as part of a group exhibition at The Sydney Fringe. To create an effective decomposition, it is important to maintain the key elements—what was required (censorship of images), the context (as part of a group exhibition), and the location/event (The Sydney Fringe).

Since the given text is already concise and clear, it is sufficient to maintain the original statement as a single claim, as breaking it further could lead to excessive fragmentation without adding clarity or distinct insights.
```

### Error Type
```
None
```

### Judgement
```
Acceptable
```

==========

### Given text
```
The smallest ocean in the world is the Arctic Ocean. It is located in the northernmost part of the Earth and is surrounded by the land masses of North America, Europe, and Asia. The Arctic Ocean covers an area of about 14.05 million square kilometers.
```

### Decomposition
```
- The smallest ocean in the world is the Arctic Ocean.
- The Arctic Ocean is surrounded by the land masses of North America.
- The Arctic Ocean is surrounded by the land masses of Europe.
- The Arctic Ocean is surrounded by the land masses of Asia.
- The Arctic Ocean covers an area of about 14.05 million square kilometers.
```

### Reasoning
```
The decomposition involves breaking down the fact that the Arctic Ocean is surrounded by North America, Europe, and Asia into three distinct sub-claims. This is an example of excessive fragmentation, as it introduces unnecessary complexity and breaks down information that should be kept together for concise understanding. This fragmentation does not add value and makes the decomposition more cumbersome.

Moreover, this fragmentation alters the original meaning. The original text implies that the Arctic Ocean is surrounded by all three continents collectively, but decomposing them into separate sub-claims can suggest an incorrect interpretation that each continent surrounds the ocean independently.

Additionally, the decomposition omits the core contextual detail that the Arctic Ocean is located in the northernmost part of the Earth. This omission reduces the completeness of the decomposition and makes it more challenging to verify the claims properly.
```

### Error Type
```
- Over-Decomposition: Excessive Fragmentation
- Over-Decomposition: Redundant Information
- Omission of Context Information: Missing Core Claims or Key Details
- Alteration of Original Meaning
```

==========

### Given text
```
As an AI language model, I cannot make subjective statements or predictions about the success or failure of a particular aircraft. However, it is worth noting that the Sukhoi Su-57 has faced some challenges in its development, including delays and budget constraints. Additionally, there are concerns about the aircraft's engine and stealth capabilities. Ultimately, the success of the Su-57 will depend on a variety of factors, including its performance in combat situations and its ability to meet the needs of the Russian military.
```

### Extracted claims
```
- I am an AI language model.
- I cannot make subjective statements.
- I cannot make predictions about the success of a particular aircraft.
- I cannot make predictions about the failure of a particular aircraft.
- The Sukhoi Su-57 has faced challenges in its development.
- The challenges faced by the Sukhoi Su-57 include delays.
- The challenges faced by the Sukhoi Su-57 include budget constraints.
- There are concerns about the aircraft's engine.
- There are concerns about the aircraft's stealth capabilities.
- The success of the Su-57 will depend on a variety of factors.
- One factor is its performance in combat situations.
- Another factor is its ability to meet the needs of the Russian military.
```

### Reasoning
```
The sub-claims about being an "AI language model" and the inability to make subjective statements or predictions are unnecessary and irrelevant for fact-checking. These elements are unverifiable and do not contribute to verifying the overall factuality of the text, as they do not present objective information. So the issue here is  **redundant information**.

The original claim contains verifiable information regarding the challenges faced by the Sukhoi Su-57, concerns about its capabilities, and factors that will determine its success. Decomposing it into too many sub-claims, such as separately listing delays, budget constraints, engine concerns, and stealth capabilities, may lead to **Excessive Fragmentation**. Since these details are interrelated and flow naturally as part of a coherent explanation, they can be grouped more effectively for simplicity and clarity.

Overall, the decomposition includes redundant information, over-fragments certain elements, and focuses on unverifiable details.
```

### Error Type
```
- Over-Decomposition: Excessive Fragmentation
- Over-Decomposition: Redundant Information
```

==========

### Given text
```
In a letter to Steve Jobs, Sean Connery refused to appear in an apple commercial.
```

### Decomposition
```
- Sean Connery refused to appear in an Apple commercial in a letter to Steve Jobs.
```

### Reasoning
```
As the original text is clear and concise enough, the provided decomposition maintains all the key details: Sean Connery's refusal, the communication method (a letter), and the context of the Apple commercial. The decomposition is self-contained and does not omit any necessary details or over-explain. Therefore, no further decomposition is needed.
```

### Error Type
```
None
```

### Judgment
```
Acceptable
```

==========

### Given text
```
President Muhammadu Buhari sign a new Police bill to allow officers arrest without a warrant
```

### Decomposition
```
- President Muhammadu Buhari signed a new Police bill.
- The new Police bill signed by Muhammadu Buhari allows officers to arrest without a warrant.
```

### Reasoning
```
The decomposition effectively splits the given text into two sub-claims: one about President Muhammadu Buhari signing a new Police bill and another specifying that the bill allows officers to arrest without a warrant. This decomposition captures both key elements of the original text clearly and concisely, without omitting any critical information or introducing ambiguity. Each sub-claim is self-contained and verifiable independently, and no unnecessary fragmentation or redundancy is present.
```

### Error Type
```
None
```

### Judgment
```
Acceptable
```

==========

### Given text
```
Sobral argued at his winners' press conference, however, that it held a humanitarian message, not political one.
```

### Decomposition
```
- Sobral won an event.
- A press conference was held.
- Sobral spoke at the press conference.
- Sobral argued a point at the press conference.
- The point Sobral argued was not a political message.
- The point Sobral argued was a humanitarian message.
```

### Reasoning
```
The decomposition excessively fragments the original text into multiple sub-claims, many of which are unnecessary and do not add significant value. 

1. Breaking down "Sobral won an event" and "a press conference was held" adds unnecessary sub-claims that do not contribute meaningfully to verifying the main point. The fact that Sobral won an event and that a press conference was held are details that can be included in a single, concise sub-claim.

2. The decomposition includes redundant sub-claims, such as "Sobral spoke at the press conference" and "Sobral argued a point at the press conference." These could be combined to convey the information in a more straightforward manner.

3. The decomposition also separates "The point Sobral argued was not a political message" and "The point Sobral argued was a humanitarian message." These two points should be kept together to maintain the full context of what Sobral argued, as separating them could lead to an alteration in meaning or loss of clarity.

Overall, the decomposition introduces unnecessary complexity through excessive fragmentation and redundancy, making the claims harder to interpret and verify.
```

### Error Type
```
- Over-Decomposition: Excessive Fragmentation
- Over-Decomposition: Redundant Information
```

### Judgment
```
Problematic
```

==========

### Given text
```
Other Countries Have Proved that the U.S. FDA Was Wrong to Withdraw Approval Of Hydroxychloroquine as COVID Drug.
```

### Decomposition
```
- Other countries have proved something.
- The U.S. FDA withdrew approval of hydroxychloroquine.
- Hydroxychloroquine was considered a COVID drug.
```

### Reasoning
```
The decomposition lacks sufficient detail to fully convey the original claim and results in ambiguity and omission of important logical relationships. "Other countries have proved something" is vague and does not communicate the specific point being argued. Additionally, the connection between the actions of other countries and the U.S. FDA's decision is lost in the decomposition, which is a key part of the claim.

The core claim involves other countries proving that the U.S. FDA made an incorrect decision about hydroxychloroquine, implying a contradiction or difference in judgment. This context should be retained to accurately reflect the claim.
```

### Error Type
```
- Omission of Context Information: Missing Core Claims or Key Details
- Omission of Context Information: Missing Logical Relationships
- Ambiguity: Vague Language
```

### Judgment
```
Problematic
```

==========

### Given text
```
{input_text}
```

### Decomposition
```
{sub_claims}
```
'''




# extract_judgment_and_improved_decomposition
def extract_sections(text):
    pattern = r"### Reasoning\n```\n(.*?)\n```\n\n### Error Type\n```\n(.*?)\n```\n\n### Judgment\n```\n(.*?)\n```\n\n### Refined Decomposition\n```\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        reasoning = match.group(1).strip()
        error_type = match.group(2).strip()
        judgment = match.group(3).strip()
        refined_decomposition = match.group(4).strip()
        
        return {
            "Reasoning": reasoning,
            "Error Type": error_type,
            "Judgment": judgment,
            "Refined Decomposition": refined_decomposition
        }
    else:
        return None

def strip_string(s: str) -> str:
  """Strips a string of newlines and spaces."""
  return s.strip(' \n')

def get_self_diagnosis_prompt(input_text: str, sub_claims: list) -> str:
    input_text = input_text.strip()
    sub_claims_str = ""
    for claim in sub_claims:
       if not claim.startswith("- "):
           claim = "- " + claim
       sub_claims_str += claim + "\n"
    sub_claims_str = sub_claims_str.strip()
    return SELF_DIAGNOSIS_PROMPT_TEMPLATE.format(input_text=input_text, sub_claims=sub_claims_str)

def extract_diagnosis(text):
    sections = extract_sections(text)
    return sections





def extract_detection_sections(text):
    pattern = r"### Reasoning\n```\n(.*?)\n```\n\n### Error Type\n```\n(.*?)\n```\n\n### Judgment\n```\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        reasoning = match.group(1).strip()
        error_type = match.group(2).strip()
        judgment = match.group(3).strip()
        
        return {
            "Reasoning": reasoning,
            "Error Type": error_type,
            "Judgment": judgment,
        }
    else:
        return None

def get_detection_prompt(input_text: str, sub_claims: list) -> str:
    input_text = input_text.strip()
    sub_claims_str = ""
    for claim in sub_claims:
       if not claim.startswith("- "):
           claim = "- " + claim
       sub_claims_str += claim + "\n"
    sub_claims_str = sub_claims_str.strip()
    return DETECTION_PROMPT_TEMPLATE.format(input_text=input_text, sub_claims=sub_claims_str)

def extract_detection(text):
    sections = extract_detection_sections(text)
    return sections

