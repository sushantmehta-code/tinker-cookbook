Let's create a new directory in recipes/ for doing RLVR and training a an open model using verifiable rewards with an LLM judge. See the details below very carefully and make a plan to do the follows:

1. use a tinker supported open model for training

2. use RLVR (RL with veirifable rubrics) for training this model

3. for the RLVR dataset: I have 1000 prompts, each prompt has several rubrics. we can use the gpt-5-mini judge model to score a model response based on how many rubrics it ended up passing. so a given response can get a score between 0 and 1 depending on the rubric pass rate. this score should be used for calculating advantages for use in a GRPO style update. the RLVR dataset is focused on instruction following, so an ideal outcome would be seeing model improvements on the hold out set after doing RL.

4. I have attached an example of 1 single prompt from the RLVR dataset in the file single_turn_IF_single_sample.json. as you can see it has the user prompt and then a bunch of rubrics (rubric - 1. criterion, rubric - 2. criterion, rubric - 3. criterion etc) that we should send to the LLM judege model to score a model response for that prompt. Note: the other rubric metadata is not important for us.

5, you should also look at CLAUDE.md to better understand this repository if needed.

5. i have an openai API key which we can use for the judge LLM calls. for efficiency purposes, to ensure we don't get bottlenecked on the LLM judge model, we should ensure use the judge model to score all rubrics for a given reponse together. so for example, if a prompt has 15 rubrics, we should make the LLM judge return a Yes/ No regarding whether the repsonse satisfies the rubric, for all rubrics together. this can then be used to calculate the rubric pass rate for that response. this should be much faster than making 15 judge llm calls for 1 response. 

6. here's an example of the LLM judge prompt that you can use for reference:

You are evaluating a response against multiple rubric criteria. Your task is to determine if the response satisfies each given criterion.

**CRITICAL: For each rubric item, you MUST output exactly one of these options: `Yes`, `No` **

Choose the option that best represents your evaluation of whether the response meets each rubric criterion.

Prompt:
{prompt}

Completion:
{completion}

Rubric items:
{rubrics}

Respond as a JSON object with a "ratings" field containing a list of objects, where each object has:
- rating: str in ["Yes", "No"]
- rationale: str explaining the rating

The list should have one entry for each rubric item in the same order as provided above.



