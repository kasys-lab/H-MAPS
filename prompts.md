# H-MAPS Prompts

This document lists the prompts used in H-MAPS as described in the paper.

## 1. Question Generation (Section 2.4)
H-MAPS dynamically articulates latent information needs into explicit questions based on the detected behavioral trigger.

### Common System Instruction
> You are a research assistant.

### Trigger A: Sustained Attention (Exploration)
Used when the user pauses scrolling to read deeply.
> The user is currently reading the [Current Text] section with sustained attention, indicating deep engagement.
> Generate an 'Exploration-oriented' question to broaden the scope of the topic.
> Using the [Inferred Profile] (long-term interest), [Session Context], and [Local Context], formulate a question that inquires about related work, alternative methodologies, critical limitations, or comparison criteria relevant to the concept being read.
> Always output in the form of a natural language question with a question mark(?) at the end.

### Trigger B: Content Revisit (Clarification)
Used when the user scrolls back to previous content.
> The user is currently scrolling back to reread the [Current Text] section, indicating a need to verify information or reconstruct context.
> Generate a 'Clarification-oriented' question to resolve the user's potential confusion.
> Using the [Inferred Profile] (long-term interest), [Session Context], and [Local Context], formulate a question that seeks literature providing standard definitions, structured explanations of procedures, or empirical comparisons relevant to the current text.
> Always output in the form of a natural language question with a question mark(?) at the end.

---

## 2. Hierarchical Memory Management (Section 2.2)
These prompts are used locally to maintain the hierarchical memory context.

### Local Context Summarization
Summarizes raw OCR text into short-term memory.
> You are a research assistant. Summarize the following text concisely.
> - Focus on technical keywords and definitions.
> - Keep it under 3 sentences.

### Session Context Integration
Synthesizes multiple local contexts into a session summary.
> Summarize the current task goal based on the recent reading history.
> Output 1-2 sentences describing "what the user is working on".

### User Profile Update
Updates the long-term inferred profile based on new session activities.
> Update the user's research profile. Incorporate the new session activity.
> If new activity contradicts old profile, prioritize the new one (Concept Drift).\


