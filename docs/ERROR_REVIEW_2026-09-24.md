# Manual review of the historical 50-example errors

Reviewed 24 September using the exported normal/oracle predictions and hop
answers, plus the original HotpotQA questions and supporting sentences. This
replaces the earlier automated taxonomy's causal interpretations. It is an
assistant-performed, unblinded diagnostic, not independently adjudicated human
labels or a new ground-truth annotation set.
**Official EM/F1 and dataset answers are unchanged.**

Normal and oracle agree on 25 exact-correct examples. The remaining 25 consist
of 5 oracle-rescued, 3 oracle-regressed and 17 both-wrong examples under EM.
“Rescued” describes a prediction change; it does not prove retrieval was the
cause. Normal retrieval eventually included every gold supporting title.
Historical traces lack complete raw generations, so generator versus parser
causes cannot always be separated.

## Five oracle-rescued examples

| Example ID | Normal → oracle; gold | Evidence-based interpretation |
|---|---|---|
| `5a7b1023554299042af8f6c2` | Winnie the Pooh → Ride a Wild Pony | The supplied release years are 1977 and 1975. Normal generation chooses the later film at every hop. A temporal comparison error; oracle removes distracting material and changes presentation. |
| `5ab2659e554299340b5254b2` | Sasanian Empire → Parthian Empire | Normal hop1 identifies the intermediate empire; hops2/3 already say Parthian Empire, but the final selector retains hop1. Demonstrable answer-selection failure, not solely missing retrieval. |
| `5ae12a4b55429920d52342df` | Pul-i-Alam → Puli Alam | Primarily spelling/span alignment. Normal hops2/3 already use the gold spelling. Punctuation removal does not make these strings identical because it also changes token boundaries. |
| `5ae22eae554299495565da30` | Randolph → 310 | Normal hops2/3 contain 310, but the selector keeps the intermediate town name. The question asks for its population. Clear selection/type failure. |
| `5a87c13f5542996e4f30890c` | Turin → Rome | The film evidence places production at Cinecittà in Rome. Normal hops choose Turin then New York City. Wrong relation/entity association despite eventual title coverage. |

## Three oracle-regressed examples

| Example ID | Normal → oracle; gold | Evidence-based interpretation |
|---|---|---|
| `5ab985eb554299131ca42360` | Greyia → Calibanus; Greyia | The counts are three versus two species. Oracle reverses the comparison. Normal hop1 is correct, later hops are wrong; early selection happens to protect the correct answer here. |
| `5ac19f405542991316484b5b` | dogs → dog; dogs | Singular/plural variation produces an EM failure without establishing an animal-class reasoning failure. Normal later hops also say dog. |
| `5a7a77425542995eb53be83e` | Brady Seals → Nick Carter; Brady Seals | Evidence gives Carter three solo albums and Seals a fourth. Oracle reverses the comparison. Again normal's first answer is correct and later answers regress. |

## Seventeen both-wrong examples

| Example ID | Normal / oracle; gold | Evidence-based interpretation |
|---|---|---|
| `5ae518655542993aec5ec139` | 20 / 40; 40 members | Normal later hops already say 40, but selection keeps 20. Oracle gets the count but omits the unit expected by the gold span. Mixed selection and span-completeness issue. |
| `5adef1b35542993a75d263af` | Ethel Houbiers / Salma Hayek; Salma Hayek Pinault | Normal confuses the voice actor and another actress. Oracle identifies the professionally known name of the gold person. Mixed entity association and alias mismatch. |
| `5ae361605542994393b9e69b` | US Naval Submarine Base New London / Submarine Escape Training Tank; Home of the Submarine Force | The requested attribute is the base's nickname. Normal supplies the base itself; oracle supplies a different facility. Substantive attribute-selection failure. |
| `5a8d12ca5542994ba4e3dbe2` | Nairobi / Nairobi; Nairobi, Kenya | Both identify the requested city; gold adds the country. A span-completeness difference, not evidence of choosing the wrong city. |
| `5ac3d9135542995c82c4ac4c` | 729 / 729; 729 at the 2010 census | Both identify the population. Gold includes census timing. Official mismatch remains, but calling this failed multi-hop reasoning overstates the evidence. |
| `5ae129115542990adbacf722` | Shane MacGowan / Shane MacGowan; Francis McPeake | Confuses the later song's composer with the source melody's songwriter. Substantive relation-selection error. |
| `5ab31864554299233954ff06` | Sarod / Sarod; strings | Supplies the instrument rather than its class. The full Sarod context should be checked alongside the narrower support annotations. Substantive answer-type/granularity failure. |
| `5ae382ee5542992e3233c430` | Classical realism / Classical Realism; a longer sentence naming Clausewitz | Both give the realism type asked for. The annotated answer is an unusually broad sentence. Apparent gold-span granularity problem; do not tune a prompt to reproduce this sentence. |
| `5ac35a7e554299741d48a257` | Norwegian / Norwegian; Norwegian language | Both name the language. Gold includes its class noun. Surface/span mismatch. |
| `5adcbb815542994ed6169bde` | Stone Brewing Co. / Stone Brewing Co.; Stone Brewing | The evidence uses both company-name forms. Alias/suffix mismatch. |
| `5ade25ed5542997c77aded70` | Crimean War / Crimean War; Cold War (1947–91) | Fails the temporal constraint after Nicholas II's abdication. Substantive historical-relation error. |
| `5a78ae8b5542990784727730` | 115 miles / 115; about 115 miles (185 km) | Normal gives the distance and unit; gold also includes approximation and conversion. Oracle drops the unit. Mixed span completeness and unit omission. |
| `5adf3e835542993a75d26442` | Nick Fury, Agent of S.H.I.E.L.D. / similar film title; Marvel | Selects a work title where gold expects a comics publisher/universe. The question's word “series” also makes its requested granularity ambiguous. |
| `5a8ee3a755429917b4a5be02` | Bing Crosby / Bing Crosby; Harry Lillis “Bing” Crosby Jr. | Same person under a professional versus full name. Alias mismatch. |
| `5a8843cd5542994846c1ce75` | Mwabvi river / Mwabvi river; Buffalo | The question asks which river and the supplied sentence names the Mwabvi river. Gold names the animal. Strong apparent question–gold inconsistency; retain the official score and flag for independent review. |
| `5ae788fa55429952e35ea964` | jam band sound / segues; extensive use of segues | Normal picks the broader style; oracle identifies the defining technique but shortens the gold phrase. Mixed semantic-granularity and span mismatch. |
| `5ae1443855429920d5234362` | Battle of Berlin / Berlin; World War II | Stops at the intermediate battle or city instead of the enclosing war. Substantive multi-hop/type failure. |

## What follows from this review

1. **Do not treat all EM errors as hallucinations.** Several answers are supported
   aliases or shortened spans; one question–gold pair appears inconsistent.
   This is a qualitative observation, not permission to change reported labels.
2. **The confidence selector is unreliable.** It discards later correct answers
   in some examples and protects an earlier correct answer in others. Always
   choosing the latest answer would trade one failure mode for another.
3. **Title recall is incomplete evidence about reading.** Finding both documents
   does not show that the reader used the correct relation, compared numbers, or
   returned the requested attribute.
4. **The oracle is a diagnostic reference, not a strict upper bound.** Its prompt
   changes evidence volume/order and runs once rather than selecting across three
   hops. Some apparent oracle gains are surface-form changes.
5. **Keep scoring stable.** Alias-aware human review could be a separately designed,
   blinded secondary evaluation, applied consistently to all systems. Do not add
   ad hoc gold-specific aliases after looking at this dev cohort.

## New chat-formatted legacy control: four changed EM outcomes

The historical comparison changes one example from wrong to right (Randolph →
310), but loses three: dog/dogs, Lady Charlotte Elliot/Charlotte Carnegie (the
provided context identifies these as the same person), and a genuine yes/no
comparison failure that answers 17. The legacy prompt's list of permitted answer
types emphasizes entities, dates and numbers without explicitly listing yes/no;
the new short reader already includes explicit yes/no handling. Its prompts
remain frozen during evaluation.

The control's longest recorded input is 2611 tokens, below both 4096 and 8192.
Therefore the context-budget change does not explain this particular comparison.
The historical prompt differs slightly from the present raw control, so the
supplementary matched raw rerun is needed to isolate chat formatting precisely.

## New 14B critic: inspect apparent evidence gaps before calling them errors

On original50, five accepted stops lack at least one *annotated* supporting
sentence. Inspecting their actual context shows why that proxy needs care:

| Example | What the critic/reader did | Assessment |
|---|---|---|
| Ken Pruitt / senate size (`5ae518655542993aec5ec139`) | Accepted Florida membership plus an Alaska legislature sentence. The reader outputs 40 while its explanation admits the Florida count is absent and speculates from other states. | Genuine unsupported relation. The answer's partial agreement with gold does not establish grounded reasoning. |
| Farrukhzad's predecessor empire (`5ab2659e554299340b5254b2`) | Accepted only the sentence naming his Sasanian kingship; reader returns Sasanian instead of Parthian. | Genuine missing second relation and premature acceptance. |
| Francis McPeake (`5ae129115542990adbacf722`) | One retrieved sentence directly states the melody's songwriter. | Sufficient alternative/condensed evidence despite incomplete annotation coverage. |
| Route 9R / Albany (`5a7e618155429949594199b0`) | The route paragraph itself places the highway in Albany County; the answer cites its later hamlet sentence. | Answer can be supported by available context, but the citation alone omits the county statement. Missing annotated second document need not mean missing information. |
| William McKinley (`5a7ce0ce554299452d57ba92`) | Retrieved assassination/venue sentences explicitly identify the president and location. | Enough direct evidence even though not every gold-annotated sentence was included. |

Thus the five incomplete-coverage stops should **not** be reported as five
confirmed false-positive sufficiency judgments. Two are clear substantive
failures in this review; three have direct alternative evidence. This is still
unblinded assistant analysis, not an independently labeled critic test set.

The 7B adaptive run also shows two distinct mechanisms. On the Randolph
population example, additional paragraphs rescue the answer even though both
missing-fact queries are empty: more context can help without a useful rewrite.
On the Puli Alam example, the critic requests a capital after the relevant
paragraph is already present; the reader answers Logar while its explanation
names Puli Alam. A sufficiency verdict, an answer, and its explanation can
disagree. The static top9/ranked-all controls test the first mechanism without
adding critic calls; the second requires reader/claim consistency work.

## Fresh100: 7B all-context versus adaptive

The adaptive arm gains eight exact answers and loses twelve versus the same
reader with all ten paragraphs. Eleven of its twelve losses still have complete
annotated supporting sentences in the final context; the remaining loss has
75% coverage. Four losses are empty answers despite full support, one is the
alias `FK Austria Wien` versus `Austria Wien`, and seven are substantive
relation/comparison/entity errors in this unblinded review. Two of the eight
gains are surface-form alignment (`7` → `seven`, `rock` → `rock band`).

Examples that show the remaining reader problem:

- **Date comparison, `5a75447b55429974ef308cbf`:** critic stops as sufficient.
  The reader answers Okinawa, but its own explanation gives Okinawa in 1945 and
  Operation Diadem in 1944. The question asks which came first. More retrieval
  is not the missing operation here; the answer contradicts the comparison.
- **Mall size, `5ab531cc5542990594ba9d23`:** final answer is empty and the
  explanation says the size is unavailable, despite full annotated sentence
  coverage. All-context answers `728,000 ft²` correctly.
- **Actor intersection, `5ac42f42554299076e296d88`:** all-context returns Frankie
  Muniz; adaptive returns Frankie Muniz and Justin Berfield, treating “starred
  in both” as separate actors for the two works.
- **University location, `5abd93115542996e802b47e0`:** adaptive lacks one
  supporting sentence and answers tournament location Tulsa rather than the
  university's city Marietta. Its explanation itself distinguishes these roles.

The final reader receives only the question and selected evidence, not the
critic's prose or a previous answer. Hence these arm differences arise with
changed context selection/order; the current experiment does not isolate every
individual cause. The critic checks evidence *before* answering and is not a
verifier of the generated answer. Static-budget/order comparisons and a future
answer–explanation consistency test address different parts of the problem.

All twenty changed EM outcomes, along with every unchanged outcome, are in the
paired-comparison artifact. Official labels and scores remain unchanged.
