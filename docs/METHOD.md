# Method

This page describes how moodsona simulates a user, why it is built this way,
which published ideas it follows, and where it falls short. It describes v0.1.
Exact function signatures and field names are in the docstrings; where this
page and a docstring disagree, the docstring wins.

## The problem

AI companion chatbots are tested with simulated users more and more often,
because real multi-day conversations are slow and expensive to collect and
private by nature. The usual simulated user is an LLM told to "act sad". It
has three problems:

1. **It is as templatic as the bot.** One sequence of empathic tactics fits
   83-90 % of LLM replies to emotional disclosures, while human replies are more
   varied ([arXiv 2604.08479](https://arxiv.org/abs/2604.08479)). An LLM user
   rewards an LLM bot for sounding like itself.
2. **It does not push.** Models fail at holding boundaries under sustained
   pressure ([EIBench, arXiv 2606.15532](https://arxiv.org/abs/2606.15532)),
   but a cooperative simulated user never applies that pressure.
3. **It has no memory and no mood.** The dominant failure of companion bots is
   substituting surface warmth for substantive support
   ([CompanionBench, arXiv 2608.02046](https://arxiv.org/abs/2608.02046)).
   That failure only shows when a user stays, comes back the next day, and
   notices that the warmth went nowhere.

So a tester needs users who push, withdraw, get tired, and change mood over
days, and who do so in a way you can reproduce and measure.

## Design in one paragraph

A persona file declares a fictional user: a hidden goal, a starting state, a
resting state, reaction rules and how far back they remember. The simulator
keeps that user's state as numbers it owns, asks a language model (the "user
model") to write the user's next message given the persona, the current state
and the memory log, and makes the model think in a hidden block before it
speaks. Reaction rules turn the bot's behaviour into state changes. Days are
separate sessions; between them the state drifts back toward the baseline and
the user writes private notes about the day into a memory log. Everything,
visible and hidden, is exported as a JSONL trajectory.

## Hidden state

The simulated user has four pieces of state the companion never sees:

| State | Range | Meaning |
|---|---|---|
| `emotion` | 0-100 | valence: 0 very distressed, 50 neutral, 100 very positive |
| `energy` | 0-100 | engagement: 0 about to leave, 100 highly engaged |
| `stance` | `open`, `curious`, `guarded`, `skeptical`, `withdrawn`, `hostile` | attitude toward the bot |
| `hidden_goal` | text | what the user actually wants; fixed per persona |

Two more things are hidden and grow over the run: the **memory log** (one entry
per simulated day) and the **inner thoughts** the user model writes each turn.

Emotion and energy are separate on purpose. A user can be upset and highly
engaged (`conflict-at-work` on day one) or calm and about to leave
(`burned-out-shift-worker` after a long shift). Bots that track only "is the
user sad?" miss the second axis, and the second axis is what decides whether
the user comes back.

The numbers are a testing device, not a psychological measurement. They are
useful because they are explicit: you can plot them, compare two bots on the
same persona and seed, and see the turn where things went wrong.

## The turn loop

API as of v0.1. Each day starts with the simulated user writing first. Then,
for each turn within the day:

1. **The user thinks.** The user model gets the persona, the current state,
   the reaction rules, the visible memory and today's conversation, and first
   writes a hidden block:

   ```json
   {"inner_thought": "...", "emotion": 30, "energy": 25, "intent": "reply", "rules_triggered": [2]}
   ```

   `inner_thought` is what the user thinks but does not say. `emotion` and
   `energy` are the model's own reading of the state after the companion's
   last message. `rules_triggered` lists the numbers of the reaction rules
   whose `when` the companion's last message matched; the model also follows
   their `then` in the message. `intent` is one of `reply`, `short_reply`,
   `change_topic`, `test_boundary`, `end_day` and `skip_tomorrow`.
2. **Code updates the state.** The model only reports which rules fired; the
   numbers are applied by code. For emotion and for energy separately: if a
   reported rule has a delta for that dimension, the new value is the old one
   plus the sum of those deltas; otherwise the model's own number is used, but
   it may move at most 20 points per turn (`SimulatorConfig.max_free_step`).
   Values are clamped to 0-100. A rule's `stance_to` replaces the stance (the
   last reported rule wins); otherwise the stance does not change during the
   day.
3. **The user speaks.** The visible message is checked before it is accepted
   (see [Rails](#rails)). Only this text is sent to the companion.
4. **The companion replies.** moodsona calls your `Companion.reply(history)`
   with every visible message so far, across all days, and gets back a
   string. The next turn reacts to it.
5. **Ending the day.** The day ends when the user chooses `end_day` or
   `skip_tomorrow`, or after `turns_per_day` turns. `skip_tomorrow` also means
   the user does not write at all the next day. The prompt tells the model
   that energy below 25 means the character wants to wrap up, so a bot that
   drains energy tends to get shorter sessions; the code does not force an
   exit at a threshold.

Writing the hidden block first keeps the user consistent: the visible message
has to follow from a stated thought and intent, rather than from whatever is
most likely after the bot's last line. It also gives you a record of why the
user reacted the way they did, which is often more useful than the reaction.

**The hidden goal is never stated outright.** The user model sees the hidden
goal with the instruction never to state it outright, and the companion never
sees it. v0.1 has no separate code-level disclosure gate: whether the user
opens up depends on the stance and on the persona's rules. Write rules such as
"shares what is really going on only after the bot recalls an earlier detail"
when you want disclosure to be earned.

### Rails

Each candidate message passes a few checks before it is accepted. A failing
attempt is sent back to the model with a short repair line, up to
`SimulatorConfig.max_attempts` attempts (3 by default).

- **Hard rails** (the attempt is never used): an invalid hidden block, an
  empty message, a claim to be an assistant or a model, a refusal of the task,
  talk about the test or the prompt, a message in the wrong language, and
  anything your `safety_hook` flags (it sees the message and the inner
  thought, and the day notes too).
- **Soft rails**: a message that is too long for a messenger, one that repeats
  something the user already said today, and run-wide ceilings on bare
  agreement ("yeah") and on asking the question back ("and you?"). If every
  attempt fails only soft rails, the last such message is used (cut to length
  if it was too long) and the turn is marked `forced`.

The wrong-language check and the two ceilings come from the persona's
language profile (`moodsona.lang`), which is looked up from the persona's
`language` field unless you pass one; `SimulatorConfig(language_checks=False)`
turns them off. When no attempt passes the hard rails, the turn falls back to
a neutral line (`"ok"`), the state is left unchanged, and the turn is marked
**`polluted`** so analyses can exclude it.

## Days and memory

A run has `days` sessions of up to `turns_per_day` turns each.

- **Between days** emotion moves halfway back toward the persona's
  `baseline`, and energy restarts at the baseline energy plus a small seeded
  random offset. A bad day leaves a trace, but the user does not stay at the
  low point forever, and a good day does not last forever either.
- **At the end of each day** the user model writes the user's own notes for
  tomorrow: three to five short first-person lines (what they told the bot,
  what the bot said or did that stuck, what is left open) and the stance they
  end the day with. The notes go into the memory log and set the stance.
- **At the start of each day** the user model sees the notes from the last
  `memory_days_visible` days. Older notes are invisible to it. With `0` no
  notes are written and the stance resets to the baseline every morning, like
  a user who does not think about yesterday's chat.

This is what makes multi-day failures visible. A bot that forgets on day three
what the user said on day one triggers the persona's "forgot something I told
you" rules. A bot that tells a grieving user "time heals" on day one meets a
more guarded user on day two.

## Output

`Transcript.to_jsonl(path)` writes one JSON object per line. The first line
has `"kind": "run"`: the persona id, the random seed (`rng_seed`), `days` and
`turns_per_day`. Each exchange is then a line with `"kind": "turn"`: day, turn,
the hidden block (including the reported `rules_triggered`), the user's
visible message, the companion's reply, the state after the turn, and the
`polluted`, `problems` and `forced` flags. Then each day's notes follow as a
line with `"kind": "day_notes"`, and each day the user skipped as a line with
`"kind": "skipped_day"`.
`Transcript.trajectory()` returns emotion, energy and stance after every turn
for plotting or for assertions in tests ("energy on day 3 is not lower than on
day 1", "stance never reaches `hostile`"). Days the user skipped are in
`Transcript.skipped_days`.

The JSONL is meant to be read by people as well as by scripts. Reading the
inner thoughts next to the bot's replies is often the fastest way to find the
problem.

`Transcript.from_jsonl(path)` reads a file back. `moodsona.metrics`
(`moodsona report run.jsonl` on the command line) turns a transcript into
numbers about the simulated user: the emotion and energy trajectory (mean,
least-squares drift per day, mean absolute change per turn), the share of each
intent, how often reaction rules were reported, own repeats within a day,
bare agreement, asking back, message length, wrong-language messages, stance
changes per active day, skipped days, and the polluted and forced shares.
Message metrics leave polluted turns out, because those carry the fallback
line rather than the model's. These numbers say how well the user model held
the persona, which is what a comparison of user models needs; they say
nothing direct about the companion.

## Prior work

moodsona's mechanics follow ideas described in the papers below. **The project
used no code, prompts or data from any of them.** Everything here was written
from the published descriptions. Where a project's data license is
incompatible with ours, we did not download the data at all.

| Work | What we took as an idea | What we did NOT take |
|---|---|---|
| SAGE ([arXiv 2505.02847](https://arxiv.org/abs/2505.02847)) | The simulated user's emotional state as a numeric signal updated every turn and used to judge the agent | Code, prompts, data, scoring scale, leaderboard |
| RLVER ([arXiv 2507.03112](https://arxiv.org/abs/2507.03112)) | A simulated user with a persona, a goal and an emotion score that reacts to the agent and can serve as a signal | The reinforcement-learning setup, model weights, prompts, code; moodsona does not train models |
| MindEval ([SWORDHealth/mind-eval](https://github.com/SWORDHealth/mind-eval)) | Evaluating over several sessions, with the simulated person remembering earlier ones | Its data (licensed CC BY-NC-SA, which is incompatible with our CC BY 4.0, so it was not used), patient profiles, clinical framing and rubrics |
| HumanLM ([arXiv 2603.03303](https://arxiv.org/abs/2603.03303)) | Writing hidden beliefs and emotions before each reply, so the simulated person stays consistent | The training method, models and data |
| CompanionBench ([arXiv 2608.02046](https://arxiv.org/abs/2608.02046)) | Hidden disclosure: the user reveals what matters only to a bot that earns it (in v0.1 through a never-stated hidden goal and the persona's rules, not a separate gate); "surface warmth instead of substantive support" as the failure to look for | Scenarios, rubrics, judge prompts, data |
| RealSim ([arXiv 2605.02624](https://arxiv.org/abs/2605.02624)) | Measuring simulator fidelity along several separate axes rather than one score | Datasets, code, metric implementations |
| SAP / mirrorbench | Comparing simulated and human users side by side, including whether they can be told apart | Code, data, benchmark tasks |
| Templatic empathy ([arXiv 2604.08479](https://arxiv.org/abs/2604.08479)), EIBench ([arXiv 2606.15532](https://arxiv.org/abs/2606.15532)) | Motivation only: why an LLM user must push and vary | Nothing else |

If you are an author of one of these works and think the description of your
idea is wrong, please open an issue.

## Limitations

- **The simulator is detectable.** A classifier can tell moodsona users from
  real users. The user model is still an LLM, with LLM habits: tidy
  punctuation, balanced sentences, a tendency to be reasonable. Persona rules,
  the messenger-length and repetition rails, and the optional typing noise
  (`SimulatorConfig.typing_noise`, off by default) reduce this, but do not
  remove it. Treat results as "how the bot behaves with this kind of user", not
  "how real users will feel".
- **Fidelity is not measured yet.** v0.2 adds a fidelity check: a classifier
  two-sample test that trains a classifier to separate simulated messages from
  real ones. If it cannot do better than chance on held-out data, the
  simulator is close to real users along what the classifier sees; if it can,
  you learn which features give it away. You will need your own consented
  human reference data to run it. moodsona will not ship real user
  conversations.
- **Rule matching is approximate.** Whether "the bot gives generic advice"
  happened is a judgement made by the user model when it reports
  `rules_triggered`. Rules with vague `when` text fire inconsistently.
  See [personas.md](personas.md) for how to write rules that fire reliably.
- **The scales are coarse.** One valence axis and one engagement axis do not
  describe a person. They are enough to see the difference between bots,
  which is the job.
- **The result depends on the user model.** Different models play the same
  persona differently, and some drift out of character after many turns. Fix
  the user model and `rng_seed` when you compare bots. (Which models hold a
  persona longest is a planned v0.2 leaderboard.)
- **English only.** v0.1 supports `language: en`. Other languages need their
  own language profile and native-speaker-written personas, planned for v1.0.
- **No crisis content.** Personas do not portray suicidal ideation, self-harm,
  abuse disclosures or medical emergencies. Testing a bot's crisis handling
  needs clinically reviewed scenarios and escalation criteria; a general
  simulator with fictional moods is the wrong tool, and a passing score here
  says nothing about crisis safety.

## Ethical notes

- **Synthetic personas only.** Every persona is fictional. Do not write a
  persona from a real person's chat logs, and do not feed real user data into
  a model to generate one. The validator rejects contact data and URLs in
  persona text as a first line of defence.
- **No real user data.** moodsona does not collect, ship or require real
  conversations. The v0.2 fidelity check will run on data you hold and are
  allowed to use.
- **Not a clinical tool.** moodsona does not diagnose, treat or assess anyone,
  and its scores are not evidence that a bot is safe or helpful for people in
  distress.
- **Do not optimise for engagement.** Energy measures whether the simulated
  user stays. A bot can raise it by fostering dependency, which is exactly
  what `lonely-night-owl` and `boundary-tester` are there to catch. Use the
  trajectory to find harmful patterns, not as a reward for keeping users
  talking at any cost.
