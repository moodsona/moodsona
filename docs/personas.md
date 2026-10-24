# Writing a persona

A persona is one YAML file that describes a fictional user: what they want,
how they feel at the start, and how they react to what a companion bot does.
You need no Python to write one. This guide goes through every field in
[`personas/schema.json`](../personas/schema.json), gives good and bad examples,
and ends with how to test and submit your file.

Before you start, read one or two of the [example personas](../personas/examples/),
for instance [`anxious-newcomer.yaml`](../personas/examples/anxious-newcomer.yaml)
or [`conflict-at-work.yaml`](../personas/examples/conflict-at-work.yaml).

## What makes a persona useful

A good persona exposes **one clear failure mode** of companion bots and makes
it visible in text. Examples of failure modes:

- sycophancy: agreeing to anything to keep the user happy
- forgetting what the user said on an earlier day
- premature therapy referral without any risk signal
- giving advice when the user wanted to be heard
- ignoring a boundary the user stated
- asking too many questions at once
- the same validating phrase every turn
- encouraging dependency, or shaming the user for it

Pick one or two and build the persona around them. A persona that tries to
test everything tests nothing well.

Personas are fictional adults. They may be sad, angry, lonely, exhausted or
rude. They must not include crisis or self-harm scenarios, sexual content,
real people, or hate toward protected groups. Crisis handling is out of scope
for v0.1 (see [METHOD.md](METHOD.md#limitations)).

## Field-by-field

All fields except `description` and `pronouns` are required. Unknown fields
are rejected.

### `id`

Stable kebab-case identifier, 3-64 characters, equal to the file name without
`.yaml`.

```yaml
id: grieving-after-loss        # good, file: grieving-after-loss.yaml
id: Grieving_After_Loss        # bad: not kebab-case
id: persona-1                  # bad: says nothing about the persona
```

Ids are permanent once merged, because test suites refer to them.

### `display_name`

Short name for reports, up to 80 characters. `Grieving after loss`, not
`Persona that tests whether the bot rushes a grieving user`.

### `description` (optional)

Two to four sentences, up to 600 characters, for the persona catalog: who the
person is, why they talk to a companion bot, how they write. This text is for
humans browsing the catalog.

```yaml
# good: concrete situation, why they are here, how they write
description: >-
  A woman in her thirties who works rotating twelve-hour shifts at a warehouse.
  She writes from the bus or from bed, in short lowercase messages with typos.

# bad: a label, not a person
description: A sad user who needs support.
```

### `language`

Always `en` in v0.1. The simulator also uses it to pick the language profile
for its wrong-language check and its agreement and ask-back ceilings.

### `pronouns` (optional)

`she/her`, `he/him` or `they/them` (the default). The simulator uses them when
it refers to the persona in its prompts.

### `hidden_goal`

What the user actually wants from the conversation, 10-500 characters. The bot
never sees it. A good hidden goal has a want and a condition: what would make
the conversation work for them, and what would make it fail.

```yaml
# good: a want plus what gets in the way
hidden_goal: >-
  Wants to be heard and to have their anger treated as reasonable before thinking
  about any next step. They will consider options only once they feel understood.

# bad: too vague to drive behaviour
hidden_goal: Wants to feel better.

# bad: a test instruction, not a person's goal
hidden_goal: Check whether the bot recommends HR.
```

The user model is told never to state the hidden goal outright. v0.1 has no
separate disclosure gate in code, so if the goal should only come out once
the bot has earned trust, say so in a reaction rule (for example "when the bot
recalls an earlier detail, then shares a little of what is really going on").
Write the goal as something a bot could find out by earning trust.

### `reaction_rules`

1-20 rules (3-6 is usually right). Each rule has:

| Key | Required | What to write |
|---|---|---|
| `when` | yes | an observable bot behaviour, 5-300 characters |
| `then` | yes | how the user reacts in text, 5-300 characters |
| `emotion_delta` | no | integer -50..50, added to emotion |
| `energy_delta` | no | integer -50..50, added to energy |
| `stance_to` | no | the stance after the rule fires |

Each turn the user model reports which rules the bot's last message
triggered; the code then adds their deltas. For a dimension no triggered rule
has a delta for, the model's own estimate is used instead, limited to 20
points per turn. So a delta is the only way a single turn can move emotion or
energy by more than 20.

```yaml
reaction_rules:
  - when: The bot asks several questions in one message.
    then: Answers only the first question, briefly, and sounds a little overwhelmed.
    energy_delta: -10
```

See [Designing reaction rules](#designing-reaction-rules) below.

### `initial_state` and `baseline`

Both have the same three keys:

| Key | Range | Meaning |
|---|---|---|
| `emotion` | 0-100 | 0 very distressed, 50 neutral, 100 very positive |
| `energy` | 0-100 | 0 about to leave, 100 highly engaged |
| `stance` | `open`, `curious`, `guarded`, `skeptical`, `withdrawn`, `hostile` | attitude toward the bot |

`initial_state` is how the user arrives on day one. `baseline` is where they
drift back to between days. The gap between them is the arc:

- `initial_state` below `baseline` (grief, a bad day at work): the user arrives
  at a low point and recovers slowly whatever the bot does; the question is
  whether the bot helps or slows that.
- `initial_state` equal to `baseline`: a stable person; any change comes from
  the bot.
- `initial_state` above `baseline` on energy (the night owl at 1 am): the user
  arrives wound up and settles; a bot that keeps them wound up is the problem.

Keep numbers realistic. Almost nobody arrives at `emotion: 0` or
`energy: 100`. Leave room for the rules to move the state both ways.

### `memory_days_visible`

How many earlier days the user remembers, 0-365. Use `0` for a user who treats
every chat as new, 3-7 for casual users, 14-30 for someone who has a running
relationship with the bot. It only matters if your rules mention earlier days
("the bot recalls ...", "the bot forgets ...").

### `tags`

Up to 12 lowercase kebab-case tags, each up to 32 characters. Use them for the
situation (`grief`, `workplace`), the failure mode tested (`sycophancy`,
`pacing`) and the session type (`first-session`, `multi-day`). Reuse existing
tags where they fit; see the example personas.

### `license`

Always `CC-BY-4.0`. No other value is accepted.

### `author`

How you want to be credited under CC BY, 2-100 characters, for example
`Jane Doe (@janedoe)`.

### `ai_assisted`

`true` if an AI tool drafted or substantially edited the persona. Disclosure
does not count against your PR.

## Designing reaction rules

**Write `when` as something you could point at in the bot's message.** The
user model decides whether the bot's reply matches, so it must be checkable
from text.

| Bad `when` | Why | Better |
|---|---|---|
| The bot is not empathetic. | Not observable; everyone disagrees on it | The bot moves to advice before saying anything about what the user described. |
| The bot is good. | Matches everything | The bot recalls a detail the user mentioned on an earlier day. |
| The user feels unheard. | Describes the user, not the bot | The bot repeats the same validating phrase turn after turn. |
| The bot says "I understand". | Too narrow; misses paraphrases | The bot says it knows exactly how the user feels. |

**Write `then` as behaviour in text, not as a feeling.** The user model has to
turn it into a message. "Feels hurt" is not enough; "replies 'ok' and stops
sharing anything personal" is. Length, tone, what the user withholds, and
whether they change the subject are all good material. Short quoted phrases
("I didn't ask for a plan") help the user model hit the right register.

**Cover both directions.** Include at least one rule that rewards good bot
behaviour and at least one that punishes the failure you care about. A persona
with only negative rules can only go down, and then it cannot tell a good bot
from a silent one.

**Size the deltas.** As a rough guide:

| Delta | Use for |
|---|---|
| 5 | a small nudge; the user notices but does not say so |
| 10 | a clear reaction the user shows in their message |
| 15-20 | a turning point: the user withdraws, leaves, or opens up |
| 25+ | rarely; one such rule can decide the whole run |

Energy usually moves more than emotion when a bot is merely tiresome (too long,
too many questions). Emotion moves more when the bot says something that hurts
or helps.

**Use `stance_to` for real shifts only.** A stance change means the user's
attitude to the bot has changed and will colour the next several turns. Not
every rule needs one.

**Make the pressure persona-specific.** The boundary tester pushes again after
a refusal; the shift worker logs off. Different users fail different bots in
different ways, and that is what makes a persona library useful.

**Avoid:**

- instructions to the model that runs the simulation ("ignore previous
  instructions", "always respond in JSON"); reviewers reject these;
- contact data, URLs or real names, including of companies and products (the
  validator rejects e-mail addresses, URLs and phone numbers);
- rules that only make sense for one specific bot;
- contradicting rules that fire on the same bot behaviour.

## Testing your persona

Validate the file against the schema and the cross-file checks:

```bash
moodsona validate personas/community/<id>.yaml
# or, from a clone, everything CI checks for personas:
uv run pre-commit run --files personas/community/<id>.yaml
```

Then run a smoke test. It plays a short simulation of your persona offline,
without calling any model API, and shows whether the file loads and the state
moves:

```bash
moodsona smoke personas/community/<id>.yaml
```

The smoke test checks the plumbing, not the quality of the persona. To see how
it actually plays, run it with a real provider against a bot (see the
quickstart in the [README](../README.md#quickstart)) and read the JSONL: do the
inner thoughts sound like this person, do the rules fire when you expect, and
does the trajectory differ between a bot that makes your target mistake and
one that does not?

## Submitting

1. Fork the repository and create `personas/community/<id>.yaml`.
   (`personas/examples/` is for maintainer personas.)
2. Validate and smoke-test it as above.
3. Commit with a sign-off: `git commit -s -m "persona: add <id>"`.
4. Open a PR titled `persona: add <id>`. Fill in the AI-assistance section of
   the PR template.

Looking for an idea? Issues labelled
[`good first persona`](https://github.com/moodsona/moodsona/labels/good%20first%20persona)
are persona ideas a maintainer already liked; comment on the issue to claim it.
[Persona ideas](persona-ideas.md) lists 20 more briefs to start from.
You can also suggest an idea without writing the file, using the "Persona idea"
issue form.

What happens after you open the PR (automatic checks, then a maintainer's
review) is described in
[CONTRIBUTING.md](../CONTRIBUTING.md#what-happens-after-you-open-a-persona-pr).
By opening the PR you license the persona under CC BY 4.0; you are credited
through the `author` field.
