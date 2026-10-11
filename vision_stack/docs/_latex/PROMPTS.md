# Prompts for keeping the report current

Five prompts for a Claude Code session opened on this repository. Each one
takes a section name, the file name under `sections/` without `.tex` (`architecture`,
`perception`, ...). Replace `<section>` and paste.

The report and the code move separately. The code changes on the mat, and the
prose describes the code as it was when you wrote it. These prompts close
that gap without having Claude write the report for you: four only read
and report, and the fifth drafts from your own answers.

| When | Prompt |
|---|---|
| After a code change or a merged PR | 1. What went stale |
| Before writing a TODO | 2. Interview me |
| After writing, before committing | 3. Check my prose |
| Every few sections | 4. Read across sections |
| A section feels done | 5. Reviewer's pass |

---

## 1. What went stale

```text
Read docs/_latex/sections/<section>.tex and every file its "% Evidence:"
and "% Source:" comments name. Then run
  git log --oneline <commit>..HEAD -- vision_stack/src vision_stack/docs
(<commit> is the last one the section was checked against; use the
section's own last commit if I give none) and read the diffs that touch
what the section covers.

Report, without editing anything:
1. Claims in the prose the repository now contradicts: quote the
   sentence, give the file:line that contradicts it, and the current value.
2. TODOs and "Think about" notes the repository can now answer, with
   where the answer is.
3. Evidence comments that point at files or names that no longer exist.
Only facts from the files; if something can't be settled from them, say so.
```

## 2. Interview me

```text
I'm writing the "<subsection>" part of docs/_latex/sections/<section>.tex.
Read its TODO and the evidence its comments name, so you know what the
repository already settles.

Then interview me, one question at a time, about what the repository
can't tell you: the decision, what else we considered, why, who made
the call (Mike, Ana or Ignacio), and what it cost. Push back when my
answer contradicts the code or is vague. Don't ask anything the files
already answer; tell me what they say instead.

After 6 to 10 questions, give me an outline of the paragraphs, each with
the facts and file it rests on. Don't write the prose; I'll write it.
```

## 3. Check my prose

```text
Check docs/_latex/sections/<section>.tex as written (prose only, not
TODOs) against the repository. For each problem, quote the sentence and
say what's wrong:
- a number, unit, part name or pin that differs from the code or docs
  (give the file:line);
- a claim with no evidence in the repo, or a word stronger than the
  evidence ("precise", "ensures", "always");
- a \ref or \citep whose label or key doesn't exist (check every
  sections/*.tex, appendices/*.tex and refs.bib);
- a term used before it's explained, or one the reader of an embedded
  systems report wouldn't know;
- typos and grammar.
List them in order of how wrong they are. Don't rewrite whole paragraphs.
Then build it alone: make -f latex.mk only S=<section>, and report any
LaTeX errors or warnings.
```

## 4. Read across sections

```text
Read every docs/_latex/sections/*.tex that has prose. Report where they
disagree with each other: the same thing given two names (Pi Zero 2 W /
the Pi / SBC), the same number given twice with different values, a
section promising a detail "in Section X" that Section X doesn't have,
or two sections explaining the same thing. Also note tense and voice
drift (we / the robot / passive). Give a short list of fixes, each
naming which section should change.
```

## 5. Reviewer's pass

```text
Read docs/_latex/sections/<section>.tex as a senior design reviewer who
knows embedded systems but not this robot. Answer:
1. In two sentences, what did the section claim?
2. Which claim would you challenge first, and what evidence would you ask for?
3. Where did you get lost?
4. What does the section promise that the report must deliver later
   (Results, Limitations)?
Be specific about paragraphs. Don't praise and don't rewrite.
```
