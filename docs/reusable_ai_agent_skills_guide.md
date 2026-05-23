# Writing Reusable Skills for Generative AI Agents

## A Novice-to-Pro, Model-Agnostic Guide

This guide is about **writing reusable skills** for generative AI agents.

It is **model agnostic**. The methods here apply whether you use Claude, Copilot, OpenAI models, Gemini, open-source models, or a routed multi-model setup.

The focus is not just on prompting a model once.
The focus is on designing **repeatable, portable, maintainable skills** that can be used across:

- multiple tasks
- multiple workflows
- multiple models
- multiple agent systems
- multiple team members

This guide includes:

- a **novice-to-pro progression**
- deep sections on **prompt engineering**, **context engineering**, and **harness engineering**
- examples of reusable skills
- anti-patterns
- templates
- activities at the end of each major section
- a **capstone project** to tie it all together

## What is a reusable skill?

A **skill** is a reusable capability an agent can perform reliably.

Examples:

- summarize a document
- extract structured fields
- classify a request
- critique a draft
- generate a test plan
- ask a clarification question
- retrieve and ground an answer in sources
- validate whether an output meets a contract
- decide whether to act, ask, stop, or escalate

A **reusable skill** is more than a good prompt.
It is usually a package of:

1. **purpose** — what the skill is for
2. **inputs** — what it expects
3. **prompt contract** — how the task is framed
4. **context contract** — what information must be provided
5. **output contract** — what it must return
6. **decision rules** — when to proceed, ask, stop, or escalate
7. **validation** — how to check whether it worked
8. **examples/tests** — how to verify portability and reliability

## Why reusable skills matter

Without reusable skills, teams often end up with:

- one-off prompts that only one person understands
- brittle agent behavior
- inconsistent outputs across models
- no easy way to test or improve behavior
- repeated reinvention of the same logic

With reusable skills, teams gain:

- consistency
- portability
- easier debugging
- easier evaluation
- faster iteration
- better collaboration
- clearer boundaries between prompt, context, and runtime logic

## A practical definition

A reusable skill should be:

- **clear** — purpose is obvious
- **bounded** — scope is limited
- **composable** — can be combined with other skills
- **testable** — success and failure are observable
- **portable** — does not depend on one model's quirks
- **maintainable** — easy to update without breaking everything

## The central idea of this guide

A reusable skill is strongest when it is engineered across three layers:

- **Prompt engineering** — what the skill tells the model to do
- **Context engineering** — what information the skill receives and how it is packaged
- **Harness engineering** — how the skill is invoked, validated, retried, routed, and composed in a larger system

A useful mental model is:

$$
\text{Reusable Skill Quality} \approx \text{Prompt Design} + \text{Context Design} + \text{Harness Design} + \text{Evaluation Discipline}
$$

## The novice-to-pro maturity model for reusable skills

### Novice

You can write a prompt that works for one task.

### Intermediate

You can turn that prompt into a reusable template with clear inputs and outputs.

### Advanced

You can design skills that work across multiple tasks and models with validation and fallback behavior.

### Pro

You can build and maintain a library of composable, tested, observable skills that power production-grade agents.

# Part I — Foundations of Reusable Skill Design

## 1. Skill vs workflow vs agent

These are related but different.

### Skill

A focused, reusable unit of capability.

Example:

- extract action items from a meeting note

### Workflow

A sequence of skills or steps.

Example:

1. summarize meeting
2. extract action items
3. assign owners
4. draft follow-up email

### Agent

A system that decides how and when to use skills, tools, memory, and policies to achieve a goal.

A good agent usually depends on good reusable skills.

## 2. What makes a skill reusable?

A reusable skill should not depend on:

- hidden assumptions
- one specific dataset shape unless declared
- one specific model's undocumented behavior
- vague output expectations
- conversational luck

It should declare:

- what it does
- what it needs
- what it returns
- what it refuses to do
- how uncertainty is handled

## 3. The anatomy of a reusable skill

Use this design structure.

### Skill name

Make it action-oriented and specific.

Examples:

- `extract_invoice_fields`
- `classify_support_ticket`
- `grounded_policy_answer`
- `critique_marketing_copy`
- `ask_for_missing_required_fields`

### Skill purpose

A one-sentence statement of what the skill accomplishes.

Example:

> Extract the required invoice metadata from semi-structured text and return a normalized record.

### Inputs

Declare exactly what the skill expects.

Examples:

- source text
- customer metadata
- policy excerpts
- allowed categories
- required output schema

### Preconditions

What must be true before the skill should run?

Examples:

- source text must be non-empty
- at least one policy document must be provided
- categories must be known in advance

### Prompt contract

How the task is described to the model.

### Context contract

What supporting information must be supplied and in what format.

### Output contract

Exactly what the skill returns.

Examples:

- JSON with required fields
- markdown bullets
- a structured decision object
- a yes/no plus evidence

### Failure behavior

What happens if the skill lacks enough information?

Examples:

- return `needs_clarification`
- ask for missing fields
- return `insufficient_context`
- escalate

### Validation rules

How do you check the result?

Examples:

- required keys exist
- no unsupported claims
- exactly one label chosen
- quotes map to source text

## 4. A reusable skill template

```text
Skill Name:
Purpose:

Inputs:
- 

Preconditions:
- 

Prompt Contract:
- 

Context Contract:
- 

Output Contract:
- 

Failure Behavior:
- 

Validation Rules:
- 

Example Inputs:
- 

Example Outputs:
- 
```

## 5. A bad skill vs a reusable skill

### Bad one-off skill

```text
Read this and tell me what matters.
```

Problems:

- no defined purpose
- no input contract
- no output contract
- no reuse boundary
- no missing-info behavior
- hard to evaluate

### Reusable version

```text
Skill Name: extract_key_points
Purpose: Extract the most decision-relevant points from a source document.

Inputs:
- source_text
- audience
- max_points

Prompt Contract:
- identify only points that materially affect decisions
- avoid stylistic observations
- do not invent facts

Output Contract:
- bullet list of up to max_points items
- each bullet must contain a concise claim and one supporting quote or paraphrase

Failure Behavior:
- if source_text is too short or lacks decision-relevant detail, return "insufficient_detail"
```

That is already far more reusable.

## 6. Skill design principles

### Principle 1: one skill, one job

A skill should do one thing well.

Bad:

- summarize document
- identify risks
- draft email
- create project plan
- convert to JSON

All in one skill is often too much.

Better:

- `summarize_document`
- `identify_risks`
- `draft_followup_email`
- `generate_project_plan`
- `format_as_json`

### Principle 2: keep inputs explicit

Do not rely on implied knowledge.

### Principle 3: define outputs as contracts

A reusable skill should be easy to consume downstream.

### Principle 4: design for ambiguity

Good skills know what to do when information is incomplete.

### Principle 5: separate policy from task

Policy and task logic should not be mixed carelessly.

### Principle 6: optimize for portability, not cleverness

Avoid fragile model-specific tricks unless absolutely necessary.

## 7. Activities for Foundations

### Activity 1: decompose a workflow into skills

Take one real workflow you use and break it into 5 to 10 separate reusable skills.
For each, write:

- name
- purpose
- inputs
- output

### Activity 2: boundedness test

Take a skill draft and ask:

- is the job too broad?
- is the output too vague?
- can another skill consume the result reliably?

### Activity 3: rewrite one-offs

Rewrite these into reusable skills:

1. "Help me with this meeting"
2. "Review this customer issue"
3. "Make this writing better"
4. "Tell me if this is allowed"

### Activity 4: contract review

For one skill, write three versions of the output contract:

- prose
- markdown bullets
- JSON schema-like description

Compare which is easiest to validate.

# Part II — Prompt Engineering for Reusable Skills

## 1. Prompting for reuse is different from prompting for a one-time answer

A one-time prompt can succeed even if it is sloppy.
A reusable skill cannot depend on luck.

When writing prompts for reusable skills, your prompt should define:

- role or operating stance
- task objective
- decision boundaries
- constraints
- output format
- quality criteria
- missing-information behavior

## 2. Prompt engineering goal: create durable behavior

A reusable skill prompt should aim for:

- repeatability
- predictability
- clarity
- low ambiguity
- easy testing

That means avoiding prompts that are overly dependent on:

- conversational context you forgot to specify
- one phrasing variation
- hidden assumptions
- stylistic vibes instead of hard constraints

## 3. A prompt contract for reusable skills

Here is a practical prompt structure.

```text
You are performing the skill: [skill_name].

Objective:
- [what the skill should accomplish]

Inputs:
- [list the inputs the skill can use]

Instructions:
- [instruction 1]
- [instruction 2]
- [instruction 3]

Boundaries:
- [what not to do]
- [when to say information is insufficient]
- [when to ask for clarification]

Output Contract:
- [required structure]

Quality Criteria:
- [what a strong result includes]
```

## 4. From novice to pro in prompt engineering for skills

### Novice: write explicit task prompts

Example:

```text
Extract the customer's account number, issue type, and requested resolution from the message below.
Return JSON only.
If a field is missing, use null.
```

Why this is reusable:

- specific task
- constrained fields
- fixed format
- missing-value behavior defined

### Intermediate: include rules and disallowed behavior

Example:

```text
Classify the ticket into exactly one category:
- billing
- technical issue
- feature request
- account access
Use only the ticket text.
Do not infer account history or previous interactions.
If the ticket contains multiple issues, choose the primary issue and explain briefly.
Return JSON only.
```

### Advanced: define behavior under uncertainty

Example:

```text
Determine whether the request qualifies under the supplied policy.
Use only the supplied policy text and case facts.
If the policy does not clearly support a conclusion, return decision = "needs_review".
Do not guess.
Distinguish evidence from interpretation.
```

### Pro: prompt for decision-making boundaries, not just content generation

Example:

```text
If a required input is missing, do not produce a final decision.
Instead, return a clarification request listing only the missing required inputs.
If the case is high impact and the evidence is ambiguous, return escalate = true.
```

That turns the prompt from content generation into skill behavior design.

## 5. Prompt components that improve reusability

### Clear naming

Use stable names for the skill and its fields.

### Explicit verbs

Prefer:

- extract
- classify
- summarize
- compare
- critique
- validate
- route
- clarify
- escalate

### Scope limits

Examples:

- use only supplied text
- produce at most 5 bullets
- choose exactly one label
- avoid external assumptions

### Output contracts

If downstream systems or skills depend on the result, output structure matters greatly.

### Failure and refusal conditions

Examples:

- insufficient information
- conflicting evidence
- unsupported task type
- out-of-scope input

## 6. Reusable prompt patterns

### Pattern 1: extraction skill

```text
Skill: extract_structured_fields

Objective:
Extract the required fields from the source text.

Inputs:
- source_text
- required_fields

Instructions:
- use only the source text
- for each required field, extract the best matching value
- if a value is absent, return null
- do not infer unavailable facts

Output Contract:
Return a JSON object with exactly the required fields.
```

### Pattern 2: classification skill

```text
Skill: classify_item

Objective:
Assign exactly one label to the input.

Inputs:
- source_text
- allowed_labels

Instructions:
- choose exactly one label from allowed_labels
- use only evidence from source_text
- if multiple labels seem plausible, choose the primary one
- provide a brief rationale

Output Contract:
{
  "label": "",
  "rationale": ""
}
```

### Pattern 3: grounded answer skill

```text
Skill: grounded_answer

Objective:
Answer the question using only the supplied sources.

Inputs:
- user_question
- sources

Instructions:
- answer only from supplied sources
- cite the source label for each substantive claim
- if sources are insufficient, state that clearly
- do not use external knowledge

Output Contract:
- answer
- evidence
- missing_information
```

### Pattern 4: clarification skill

```text
Skill: ask_for_missing_required_fields

Objective:
Ask the smallest possible clarification needed to proceed.

Inputs:
- task_goal
- required_fields
- provided_fields

Instructions:
- identify only fields that are truly required and missing
- ask one concise clarification message
- do not ask for optional information
- do not continue with final output

Output Contract:
- clarification_message
- missing_fields
```

### Pattern 5: critique skill

```text
Skill: critique_draft

Objective:
Identify the most important weaknesses in the draft relative to the stated goal.

Inputs:
- draft_text
- goal
- criteria

Instructions:
- focus on the highest-impact weaknesses
- tie each critique to the criteria
- avoid trivial stylistic nitpicks unless they materially affect the goal

Output Contract:
- top_issues
- suggested_fixes
```

## 7. Prompt anti-patterns for reusable skills

### Anti-pattern: role-only prompts

```text
You are an expert consultant. Help with this.
```

Problem: role is not enough. The skill still lacks objective, boundaries, and output contract.

### Anti-pattern: giant all-purpose prompts

Trying to build one master prompt for every situation usually destroys reusability.

### Anti-pattern: hidden schema assumptions

If your skill expects fields like `customer_id`, `plan_name`, or `decision_reason`, declare them explicitly.

### Anti-pattern: vague quality guidance

"Be insightful" is not a strong contract.

### Anti-pattern: no missing-info behavior

A reusable skill must know what to do when it cannot finish correctly.

## 8. Example skill definitions with prompts

### Example A: `extract_meeting_actions`

#### Purpose
Extract action items, owners, and deadlines from meeting notes.

#### Prompt
```text
You are performing the skill: extract_meeting_actions.

Objective:
Extract action items from the meeting notes.

Inputs:
- meeting_notes

Instructions:
- identify only explicit or strongly implied action items
- for each action item, extract owner if present
- for each action item, extract deadline if present
- if owner or deadline is not present, use null
- do not invent commitments not supported by the notes

Output Contract:
Return JSON with this shape:
{
  "actions": [
    {
      "action": "",
      "owner": null,
      "deadline": null
    }
  ]
}
```

#### Why it is reusable

- bounded task
- structured output
- clear missing-field policy
- no dependence on one meeting type

### Example B: `grounded_policy_decision`

#### Purpose
Determine whether a case meets a policy rule using only supplied evidence.

#### Prompt
```text
You are performing the skill: grounded_policy_decision.

Objective:
Determine whether the case qualifies under the policy.

Inputs:
- policy_text
- case_facts

Instructions:
- use only policy_text and case_facts
- do not rely on outside knowledge
- if the evidence clearly supports a conclusion, return approved or denied
- if evidence is incomplete or ambiguous, return needs_review
- quote the most relevant policy language

Output Contract:
{
  "decision": "approved|denied|needs_review",
  "reason": "",
  "evidence": [""],
  "missing_information": []
}
```

## 9. Prompt engineering checklist for reusable skills

Before finalizing a skill prompt, ask:

- Is the skill purpose obvious?
- Is the task singular and bounded?
- Are inputs explicitly named?
- Are constraints clear?
- Is output structure fixed enough for reuse?
- Is missing-information behavior defined?
- Does the prompt avoid model-specific tricks where possible?
- Can I test it with multiple examples?

## 10. Activities for Prompt Engineering

### Activity 1: turn a task into a reusable skill prompt

Take a task you do often and write a prompt contract with:

- objective
- inputs
- instructions
- boundaries
- output contract
- quality criteria

### Activity 2: prompt hardening

Choose one skill and add rules for:

- uncertainty
- insufficient context
- ambiguous cases
- refusal behavior
- clarification behavior

### Activity 3: portability rewrite

Take a prompt that relies on model-specific phrasing and rewrite it so it depends more on structure than magic wording.

### Activity 4: one skill, one job test

Take one overly broad prompt and split it into at least three reusable skills.

### Activity 5: create a mini prompt library

Write prompts for these skills:

- summarize_for_executive
- classify_ticket
- extract_entities
- ask_for_clarification
- critique_draft

# Part III — Context Engineering for Reusable Skills

## 1. Why context engineering matters for reusable skills

A prompt can be reusable only if the skill can reliably consume inputs from different situations.
That means the context must also be reusable.

A skill often fails not because the prompt is wrong, but because the context was:

- incomplete
- noisy
- unstructured
- stale
- contradictory
- unlabeled

## 2. Context engineering goal: give the skill the minimum sufficient context

For reusable skills, context should be:

- relevant
- structured
- labeled
- portable
- easy to validate

Good context engineering makes a skill robust across different cases.

## 3. The context contract

Every reusable skill should declare its context needs.

Examples:

### For `classify_support_ticket`

Needs:

- ticket_text
- allowed_categories

Does not need:

- full CRM history unless classification depends on it

### For `grounded_policy_answer`

Needs:

- user question
- policy excerpts
- case facts
- source labels

### For `draft_followup_email`

Needs:

- summary of prior interaction
- desired outcome
- audience
- tone constraints

## 4. Reusable context patterns

### Pattern 1: labeled blocks

```text
[User Goal]
...

[Primary Input]
...

[Business Rules]
...

[Relevant Metadata]
...
```

### Pattern 2: fact sheet plus raw source

```text
[Fact Sheet]
- customer_tier: enterprise
- issue_type: duplicate billing
- disputed_amount: 2400

[Source Message]
...
```

This helps the skill quickly find critical facts without losing the source.

### Pattern 3: retrieved snippets with provenance

```text
[Source A: Refund Policy, updated 2026-01-10]
...

[Source B: Customer Email]
...
```

### Pattern 4: state summary for multi-step skills

```text
[State Summary]
- goal: prepare policy decision memo
- completed: gathered customer statement, retrieved policy
- open_question: whether exception approval applies
```

## 5. From novice to pro in context engineering for skills

### Novice: include the needed text

At this stage, success comes from supplying the actual source material and not forcing the model to guess.

### Intermediate: label and separate context types

Separate:

- instructions
- evidence
- metadata
- prior state

### Advanced: create context contracts per skill

For each skill, write down:

- required fields
- optional fields
- disallowed or irrelevant fields
- freshness requirements
- provenance requirements

### Pro: design composable context objects

At a professional level, teams often standardize reusable context shapes.

Examples:

- `task_context`
- `source_bundle`
- `case_facts`
- `state_summary`
- `validation_metadata`

This makes skill composition much easier.

## 6. Context anti-patterns for reusable skills

### Anti-pattern: dump everything

More context is not automatically better.
Too much irrelevant text makes skills less reliable.

### Anti-pattern: mixing instructions and facts

If the skill cannot tell what is a rule versus what is evidence, quality drops.

### Anti-pattern: missing provenance

If a skill returns evidence, it should know where that evidence came from.

### Anti-pattern: stale context without labeling

Outdated information can produce wrong outputs that look credible.

### Anti-pattern: inconsistent field naming across skills

If one skill expects `account_id` and another expects `customer_account_number` for the same concept, reuse becomes harder.

## 7. Designing context interfaces for skills

Think of each skill as having an interface.

Example:

### Skill: `evaluate_exception_request`

#### Required context

- policy_excerpt
- request_details
- exception_criteria

#### Optional context

- historical_precedent
- account_tier

#### Must not rely on

- hidden chat history
- unstated company norms
- general web knowledge

That is useful because it tells other people and other systems exactly how to call the skill.

## 8. Examples of reusable context design

### Example A: `summarize_incident`

#### Good context

```text
[Incident Timeline]
...

[Impact Summary]
...

[Known Root Cause]
...

[Audience]
Executives
```

#### Why good

- separates evidence types
- includes audience
- avoids irrelevant engineering chatter if not needed

### Example B: `extract_contract_risks`

#### Good context

```text
[Document Excerpt]
...

[Risk Categories]
- indemnity
- liability cap
- auto-renewal
- data protection

[Buyer Preferences]
- prefer mutual obligations
- avoid uncapped confidentiality exposure
```

### Example C: `ask_for_missing_required_fields`

#### Good context

```text
[Task Goal]
Create a renewal quote

[Required Fields]
- customer_name
- plan
- renewal_date

[Provided Fields]
- customer_name
- plan
```

This lets the skill ask only for `renewal_date`.

## 9. Context engineering checklist for reusable skills

- Does the skill have a declared context contract?
- Are required and optional inputs separated?
- Are sources labeled?
- Are facts separated from instructions?
- Are stale or uncertain sources marked?
- Is irrelevant context removed?
- Are field names consistent across related skills?
- Can another person supply the context without guesswork?

## 10. Activities for Context Engineering

### Activity 1: context contract writing

Choose 5 candidate skills and write for each:

- required inputs
- optional inputs
- irrelevant inputs
- freshness needs
- provenance needs

### Activity 2: refactor a messy context package

Take a real prompt and reorganize the context into:

- goal
- evidence
- rules
- metadata
- state

### Activity 3: minimal sufficient context drill

For a skill you use often, create two versions of context:

- minimal sufficient
- bloated noisy

Compare the likely failure modes of each.

### Activity 4: field naming normalization

List all fields used across 3 skills in one workflow.
Normalize naming so the same concept has the same name everywhere.

### Activity 5: provenance audit

Take one skill output and annotate each factual claim with the exact source block that supported it.

# Part IV — Harness Engineering for Reusable Skills

## 1. What harness engineering means here

For reusable skills, harness engineering is the design of how a skill is:

- invoked
- supplied with inputs
- validated
- retried
- composed with other skills
- observed
- versioned
- safely stopped or escalated

A reusable skill that works in isolation but fails in a real system is incomplete.

## 2. Harness engineering goal: make skills dependable in systems

Even a good prompt and good context can fail if the harness is weak.

Examples of harness failures:

- a skill runs without required inputs
- a skill is called when a clarification should happen first
- output is malformed and downstream steps break
- the wrong skill is routed for the task
- the system keeps retrying instead of escalating

## 3. The harness contract for a reusable skill

Each skill should ideally define:

- how it is selected
- its required inputs
- pre-run checks
- post-run validation
- retry rules
- escalation rules
- logging fields
- version identifier

## 4. From novice to pro in harness engineering for skills

### Novice: call a skill manually with explicit inputs

At this stage, the key is simply consistency.

### Intermediate: add precondition checks and output validation

Before calling the skill:

- confirm required fields exist

After calling the skill:

- validate required output structure

### Advanced: create routing and failure policies

Examples:

- if required fields are missing, call clarification skill first
- if confidence is low, escalate
- if output fails schema validation, retry once with corrective instruction

### Pro: build a skill runtime pattern

At the pro level, reusable skills often behave like callable modules with:

- versioned definitions
- standardized interfaces
- test fixtures
- telemetry
- rollback paths
- cross-model evaluation

## 5. Reusable harness patterns

### Pattern 1: precheck -> run -> validate

This is one of the most useful patterns.

1. precheck inputs
2. run skill
3. validate output
4. retry or fail gracefully if needed

Example:

#### Skill
`extract_invoice_fields`

#### Precheck
- source_text is not empty
- required_fields list is present

#### Validate
- all required keys exist
- all values are scalar or null

### Pattern 2: ask-before-finalize

1. detect missing required inputs
2. call clarification skill
3. wait for answer
4. continue with main skill

Useful for many business workflows.

### Pattern 3: draft -> critique -> revise

1. run generation skill
2. run critique skill
3. run revision skill

Useful for writing and reasoning tasks.

### Pattern 4: retrieve -> ground -> answer

1. retrieve sources
2. package context with provenance
3. call grounded-answer skill
4. validate citation/evidence use

### Pattern 5: route-by-intent

1. classify incoming request
2. choose skill based on request type
3. run selected skill with correct contract

## 6. Skill composition

Reusable skills become truly powerful when composed.

### Example workflow: customer support assistant

Skills:

1. `classify_ticket`
2. `extract_required_case_fields`
3. `ask_for_missing_required_fields`
4. `retrieve_policy_snippets`
5. `grounded_policy_decision`
6. `draft_customer_response`
7. `validate_response`

Each skill is reusable by itself, but together they form a stronger system.

## 7. Validation in harness engineering

If outputs are meant to be reused, validation is essential.

What to validate:

- output structure
- completeness
- type correctness
- policy compliance
- source grounding
- exact label constraints
- no unsupported actions

### Example validation checklist for `classify_ticket`

- exactly one allowed label selected
- rationale is present
- label belongs to allowed set
- no extra fields unless allowed

### Example validation checklist for `grounded_policy_decision`

- decision is one of approved/denied/needs_review
- reason is non-empty
- evidence array exists
- no evidence quote absent from context
- missing_information included if decision is needs_review

## 8. Retry and recovery rules

A reusable skill should not retry blindly.

Define:

- when a retry is allowed
- what changes on retry
- when to stop retrying
- when to escalate

Good retry examples:

- retry if JSON formatting failed
- retry if one required field is missing
- retry with narrowed context if output is noisy

Bad retry examples:

- retry endlessly on ambiguous policy questions
- retry without changing anything

## 9. Observability for reusable skills

If a skill is part of a shared system, log enough to diagnose failures.

Recommended log fields:

- skill_name
- skill_version
- input_summary
- prompt_version
- context_summary
- validation_result
- retry_count
- final_status
- latency
- model_used

## 10. Versioning reusable skills

A reusable skill library benefits greatly from versioning.

Examples:

- `classify_ticket:v1`
- `classify_ticket:v2`
- `grounded_policy_decision:v3`

Version when you change:

- output shape
- instructions materially
- decision boundaries
- validation rules
- context requirements

## 11. Harness anti-patterns for skills

### Anti-pattern: no precondition checks

A skill should not be called with missing required inputs unless its job is to detect that.

### Anti-pattern: skills that return inconsistent shapes

Downstream composition becomes painful.

### Anti-pattern: validation only by human eyeballing

That does not scale.

### Anti-pattern: hidden routing logic

If skill selection rules are implicit, errors become hard to debug.

### Anti-pattern: unversioned prompt edits

Small prompt changes can silently break workflows.

## 12. Example reusable skill specs with harness notes

### Example A: `classify_ticket`

#### Harness notes

- run only after `ticket_text` is present
- validate label against allowed set
- if label confidence is low and message mentions outage, escalate

### Example B: `draft_followup_email`

#### Harness notes

- run only after summary and desired outcome are present
- validate word count and required call-to-action
- if audience is missing, call clarification skill first

### Example C: `grounded_policy_decision`

#### Harness notes

- run only after at least one authoritative policy source is retrieved
- reject result if evidence quotes cannot be matched to retrieved sources
- escalate if decision is high impact and confidence is low

## 13. Harness engineering checklist for reusable skills

- Are preconditions explicit?
- Is there input validation?
- Is output validation defined?
- Are retries limited and purposeful?
- Are escalation rules documented?
- Is skill selection/routing visible?
- Is the skill versioned?
- Are logs sufficient for diagnosis?
- Can the skill be composed predictably with others?

## 14. Activities for Harness Engineering

### Activity 1: write a runtime contract

For 3 skills, write:

- preconditions
- validation rules
- retry rules
- escalation rules
- logging fields

### Activity 2: compose a workflow

Choose a real use case and map at least 5 reusable skills into a workflow.
Indicate exactly what each skill consumes and returns.

### Activity 3: validation drill

Take one skill and design a validator that checks:

- output shape
- completeness
- no unsupported fields
- rule compliance

### Activity 4: failure mapping

List 10 failure cases and label each as primarily:

- prompt failure
- context failure
- harness failure
- evaluation failure

### Activity 5: versioning exercise

Create `v1` and `v2` of a skill spec and explain what changed and why.

# Part V — Designing a Reusable Skill Library

## 1. Why think in libraries?

Once you have more than a handful of agent behaviors, you are no longer just writing prompts.
You are building a **skill library**.

A skill library helps you:

- avoid duplication
- standardize patterns
- share work across teams
- test behaviors consistently
- improve skills independently

## 2. Recommended categories of reusable skills

### Understanding skills

- classify
- extract
- detect
- summarize
- compare
- cluster

### Reasoning skills

- evaluate against policy
- identify risks
- critique draft
- propose options
- prioritize issues

### Interaction skills

- ask for clarification
- explain a decision
- draft a response
- convert technical detail for a target audience

### Control skills

- validate output
- decide ask vs act
- escalate if uncertain
- determine whether context is sufficient

### Transformation skills

- convert to JSON
- rewrite for tone
- normalize fields
- transform notes into action items

## 3. Naming conventions for skill libraries

Use names that are:

- specific
- verb-led when possible
- stable
- readable across teams

Examples:

- `extract_customer_entities`
- `classify_support_intent`
- `grounded_answer_from_sources`
- `ask_for_missing_required_fields`
- `validate_structured_output`

Avoid vague names like:

- `assistant_help`
- `smart_analysis`
- `do_task`

## 4. Standard metadata for each skill

Each skill in a library should ideally include:

- name
- version
- owner
- purpose
- input schema
- output schema
- prompt template
- context contract
- validation rules
- example cases
- known limitations

## 5. Skill granularity

Too coarse:

- hard to reuse
- hard to test
- hard to diagnose

Too fine:

- too many tiny skills with orchestration overhead

A good rule:

A skill should be large enough to be meaningful, but small enough to have a clear contract.

## 6. Example skill cards

### Skill Card: `classify_support_intent`

- Purpose: assign one primary support category to a customer message
- Inputs: `ticket_text`, `allowed_labels`
- Output: `label`, `rationale`
- Failure behavior: return `unsupported_input` if text is empty
- Validation: label must be in allowed set
- Known limitation: can compress multi-issue tickets into one primary label only

### Skill Card: `ask_for_missing_required_fields`

- Purpose: ask the minimum clarification needed to continue a workflow
- Inputs: `task_goal`, `required_fields`, `provided_fields`
- Output: `clarification_message`, `missing_fields`
- Validation: only missing required fields may be requested
- Known limitation: does not prioritize optional nice-to-have fields

### Skill Card: `grounded_answer_from_sources`

- Purpose: answer a question using only provided sources
- Inputs: `question`, `sources`
- Output: `answer`, `evidence`, `missing_information`
- Validation: claims must map to sources
- Known limitation: depends on source quality and retrieval quality

## 7. Activities for Skill Libraries

### Activity 1: build a starter library

Design 10 reusable skills for one domain.
For each, write a one-paragraph skill card.

### Activity 2: skill deduplication

List all prompts you currently use repeatedly.
Group them into candidate reusable skills and merge overlaps.

### Activity 3: granularity review

Take 5 skills and ask whether each should be:

- split
- kept as-is
- combined with another skill

### Activity 4: naming cleanup

Rename vague skill names into precise, reusable names.

# Part VI — Testing and Evaluating Reusable Skills

## 1. Why evaluation matters

A reusable skill is only reusable if you can trust it across cases.

Evaluation should test:

- correctness
- consistency
- portability across models
- robustness to edge cases
- behavior under missing information
- schema compliance

## 2. Test reusable skills like components

Each skill should have examples such as:

- happy path
- ambiguous input
- missing required data
- contradictory evidence
- adversarial or misleading phrasing

## 3. What to measure

Depending on the skill, useful measures include:

- exact field accuracy
- label accuracy
- schema validity rate
- unsupported-claim rate
- clarification quality
- escalation correctness
- citation/evidence faithfulness

## 4. Portability testing across models

Because this guide is model agnostic, test your skills across at least two model families when possible.

Look for:

- output contract stability
- sensitivity to wording
- over-inference behavior
- formatting reliability
- performance on ambiguous cases

## 5. Evaluation template

```text
Skill Name:
Version:

Test Case ID:
Input:
Expected Behavior:
Actual Behavior:
Failure Type:
Notes:
```

## 6. Common evaluation failures

### Failure: works only with one exact prompt phrasing

That indicates poor reusability.

### Failure: output format drifts across models

That indicates output contract and validation need strengthening.

### Failure: skill guesses instead of clarifying

That indicates uncertainty behavior is under-specified.

### Failure: evidence claims are unsupported

That indicates grounding or validation needs work.

## 7. Activities for Testing and Evaluation

### Activity 1: create a 15-case test set

Choose one reusable skill and create:

- 5 normal cases
- 4 ambiguous cases
- 3 incomplete-input cases
- 3 adversarial cases

### Activity 2: portability comparison

Run the same skill design against two different model families and compare:

- correctness
- structure compliance
- clarification behavior

### Activity 3: failure postmortem

For 5 failing cases, record whether the fix belongs in:

- prompt
- context
- harness
- validator

### Activity 4: contract stress test

Intentionally vary input order, wording, and metadata volume.
See whether the skill still behaves consistently.

# Part VII — Capstone Project: Build a Reusable Skill Pack

## Project goal

Create a **model-agnostic reusable skill pack** for a policy-aware support or operations assistant.

The objective is not just to build one agent.
The objective is to build a **set of reusable skills** that can power multiple workflows.

## Choose a domain

Pick one:

- support ticket handling
- refund review
- HR policy questions
- contract review triage
- internal IT helpdesk
- operations SOP assistant

## Required reusable skills

Build at least these 8 skills:

1. `classify_request_type`
2. `extract_required_case_fields`
3. `ask_for_missing_required_fields`
4. `retrieve_relevant_policy_context`
5. `grounded_policy_decision`
6. `draft_user_facing_response`
7. `validate_structured_output`
8. `escalate_if_high_risk_or_ambiguous`

You may add more if useful.

## Deliverables

### 1. Skill catalog

For each skill, provide:

- name
- purpose
- inputs
- preconditions
- prompt contract
- context contract
- output contract
- failure behavior
- validation rules
- examples

### 2. Prompt package

Write the full prompt contract for each required skill.
Make each one reusable and bounded.

### 3. Context contracts

Define the exact context interface for each skill.
Use consistent field naming across the skill pack.

### 4. Harness design

Design a workflow showing how the skills compose.

Example flow:

1. classify request
2. extract fields
3. ask clarification if needed
4. retrieve policy
5. make grounded decision
6. escalate if needed
7. draft response
8. validate output

### 5. Validation layer

Create validators for at least 4 of the skills.

### 6. Evaluation set

Create at least 20 test cases covering:

- straightforward cases
- ambiguous cases
- incomplete cases
- contradictory evidence
- adversarial wording

## Suggested skill card format

```text
Skill Name:
Version:
Purpose:

Inputs:
- 

Preconditions:
- 

Prompt Contract:
- 

Context Contract:
- 

Output Contract:
- 

Failure Behavior:
- 

Validation Rules:
- 

Examples:
- 

Known Limitations:
- 
```

## Suggested output schema example for `grounded_policy_decision`

```json
{
  "decision": "approved|denied|needs_review",
  "reason": "",
  "evidence": [
    {
      "source": "",
      "quote": ""
    }
  ],
  "missing_information": [],
  "confidence": "low|medium|high",
  "escalate": false
}
```

## Capstone review questions

Use these to assess your project:

1. Are the skills clearly separated by responsibility?
2. Can each skill be called independently?
3. Are context needs explicit and consistent?
4. Are output contracts stable enough for composition?
5. Does the harness know when to ask, act, stop, and escalate?
6. Are validators catching common failure modes?
7. Would these skills still work if you swapped to another major model?
8. Which skill is most brittle and why?
9. Which skill is most reusable across other workflows?
10. What library standard would you add next?

# Part VIII — Quick Reference Cheat Sheets

## Reusable skill design cheat sheet

- One skill, one job.
- Name it clearly.
- Declare inputs explicitly.
- Define preconditions.
- Specify prompt contract.
- Specify context contract.
- Specify output contract.
- Define failure behavior.
- Add validation.
- Add examples and tests.

## Prompt engineering cheat sheet for reusable skills

- Write for durability, not one-time brilliance.
- Use explicit verbs.
- Define boundaries.
- Define missing-info behavior.
- Use fixed output structures.
- Avoid model-specific magic where possible.

## Context engineering cheat sheet for reusable skills

- Provide minimum sufficient context.
- Label context blocks.
- Preserve source provenance.
- Separate rules from evidence.
- Normalize field names.
- Mark stale information.

## Harness engineering cheat sheet for reusable skills

- Precheck before running.
- Validate after running.
- Limit retries.
- Route explicitly.
- Escalate when appropriate.
- Log key metadata.
- Version your skills.

# Final Advice

If your goal is to get good at **writing reusable skills** for generative AI agents, focus less on finding a magical prompt and more on designing a **portable contract**.

The highest-value habit you can build is this:

1. define the skill clearly
2. define what context it needs
3. define what it returns
4. define what happens when it cannot finish safely or correctly
5. validate it with realistic test cases
6. version and improve it over time

That is how one-off prompting turns into reusable agent engineering.

## Suggested next step

Pick one recurring task from your work this week and turn it into a full skill card with:

- purpose
- inputs
- prompt contract
- context contract
- output contract
- failure behavior
- validation rules
- 5 test cases

Do that repeatedly, and you will start building not just prompts, but a real skill library for agents.
