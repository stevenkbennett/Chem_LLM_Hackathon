# Task 1 
1.	train a model with varying batch size, beam size, epoch, training set 
1.	try augmenting the training set
2.	see what score is lowest - duplicates and invalid smile
3.	Post-Processing Deduplication: The simplest method would be to introduce a deduplication step in your post-processing pipeline. After the model has generated the top 10 predictions, you can programmatically eliminate duplicates, replacing them with the next best unique prediction from the model.
4.	Post-processing remove invalid smile
5.	post process – sort results
6.	Use the randomness of the system, try many times to obtain the best result
1.	re-run the test set through the LLM

 
# Task 2
Strategy
@ deeplearning.ai 
## Prompt Engineering - Principles
Write clear and specific instructions
Give the model time to “think”
Principle 1
Tactic 1: Use delimiters to clearly indicate distinct parts of the input
Delimiters can be anything like: ```, """, < >, <tag> </tag>, :
Tactic 2: Ask for a structured output
Tactic 3: Ask the model to check whether conditions are satisfied
Tactic 4: "Few-shot" prompting
Give successful examples of completed tasks before asking the model to perform the task
Principle 2
Tactic 1: Specify the steps required to complete a task. Ask for output in a specified format
Tactic 2: Instruct the model to work out its own solution before rushing to a conclusion
Model Limitations: Hallucinations

	Copy and paste the question but rewrite it a bit more specific and ask it to give a “detailed scientific answer including facts and numbers in a profession language”
	The answer usually too long, so I asked it to “Concise it down to less than 100 words in a scientific way “
	If the answer score low, try to specifically ask it to elaborate in a more in-depth details from its prior answer on a specific point.
	Then concise it down again and keep on doing until you get an improved answer.
	If nothing improves, I either tried refresh ChatGPT and start over again, so it will start with a new set of answer that might be better or worse. Then, repeat the process.
	Or use other software e.g., Claude (I found it kind of give more numbers than ChatGPT) and then work with that answer on ChatGPT to concise it down etc.
	Usually the more concise answer with some comparisons with details score higher than longer redundant answers.


## Prompt histories
https://chat.openai.com/share/d3bf31e5-08a8-43d1-900d-536f2a51506f
https://chat.openai.com/share/860f864c-7241-4377-a50c-da322aa23f57
https://chat.openai.com/share/1377a844-e400-496e-b4bc-1074246af0e3
https://chat.openai.com/share/54a134c2-7cdd-4f71-a26c-af68a1df9288

## Prompt histories 2
Here are 5 questions to answer below: 

1. What are the key differences between conductors, semiconductors and insulators?
2. What does Bloch's theorem state about electron waves in periodic lattices?
3. How does one go from atomic displacements to phonons?
4. What is the key feature of the band structure in graphene?
5. What is the Pauli exclusion principle?

Task: Answer the physics exam questions above in the style of a physicist. Use no more than 100 words per question. 
Output: An JSON array. Each JSON should have the following keys: 
- id: which should be the numeric number assigned to the question above; 
- answer: the answer for the question above, answer in a style of a physicist. Write no more than 100 words. ;Here’s an example:
Input:
1. What is fourier transformation? 
2. Describe the doppler's effect. 
Output:
[
   { "id": 1,"explanation":"In physics and mathematics, the Fourier transform (FT) is a transform that converts a function into a form that describes the frequencies present in the original function. The output of the transform is a complex-valued function of frequency. The term Fourier transform refers to both this complex-valued function and the mathematical operation. When a distinction needs to be made the Fourier transform is sometimes called the frequency domain representation of the original function. The Fourier transform is analogous to decomposing the sound of a musical chord into terms of the intensity of its constituent pitches."},
   { "id": 2,"explanation":"Doppler Effect refers to the change in wave frequency during the relative motion between a wave source and its observer. It was discovered by Christian Johann Doppler who described it as the process of increase or decrease of starlight that depends on the relative movement of the star."}
]
DO NOT include anything other than the JSON array in your response.



---------------------------------------------------------

Here are 5 questions to answer below: 

3. Carbon monoxide is a good ligand. Why is the isoelectronic N2 molecule not a good ligand?
4. Describe the 1H NMR spectrum of GeH4.

Task: Answer the chemistry exam questions above in the style of a chemist. Use no more than 100 words per question. 
Output: An JSON array. Each JSON should have the following keys: 
- id: which should be the numeric number assigned to the question above; 
- answer: the answer for the question above, answer in a style of a chemist. Write no more than 100 words. ;Here’s an example:
Input:
1. What is covalent bond? 
2. What are characteristics of group 10 elements? 
Output:
[
   { "id": 1,"explanation":"A covalent bond is a chemical bond that involves the sharing of electrons to form electron pairs between atoms. These electron pairs are known as shared pairs or bonding pairs. The stable balance of attractive and repulsive forces between atoms, when they share electrons, is known as covalent bonding.[1] For many molecules, the sharing of electrons allows each atom to attain the equivalent of a full valence shell, corresponding to a stable electronic configuration."},
   { "id": 2,"explanation":"Group 10 metals are white to light grey in color, and possess a high luster, a resistance to tarnish (oxidation), are highly ductile, and enter into oxidation states of +2 and +4, with +1 being seen in special conditions."}
]

DO NOT include anything other than the JSON array in your response.
