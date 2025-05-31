from sklearn.metrics import accuracy_score, classification_report

class HMM:
    def __init__(self):
        self.transition_probs = {}
        self.emission_probs = {}
        self.tag_counts = {}
        self.vocab = set()
        self.tags = set()
        self.start_tag = "<START>"
        self.end_tag = "<END>"

    def train(self, tagged_sentences):
        transitions = {}
        emissions = {}
        tag_counts = {}

        for sentence in tagged_sentences:
            prev_tag = self.start_tag
            tag_counts[prev_tag] = tag_counts.get(prev_tag, 0) + 1
            for word, tag in sentence:
                self.vocab.add(word.upper())
                self.tags.add(tag)

                # Emissions
                emissions.setdefault(tag, {})
                emissions[tag][word.upper()] = emissions[tag].get(word.upper(), 0) + 1

                # Transitions
                transitions.setdefault(prev_tag, {})
                transitions[prev_tag][tag] = transitions[prev_tag].get(tag, 0) + 1

                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                prev_tag = tag

            # Transition to end tag
            transitions.setdefault(prev_tag, {})
            transitions[prev_tag][self.end_tag] = transitions[prev_tag].get(self.end_tag, 0) + 1
            tag_counts[self.end_tag] = tag_counts.get(self.end_tag, 0) + 1

        # Calculate probabilities
        self.transition_probs = {
            prev_tag: {tag: count / tag_counts[prev_tag] for tag, count in tags.items()}
            for prev_tag, tags in transitions.items()
        }
        self.emission_probs = {
            tag: {word: count / tag_counts[tag] for word, count in words.items()}
            for tag, words in emissions.items()
        }
        self.tag_counts = tag_counts

    def viterbi(self, sentence):
        sentence = [w.upper() for w in sentence]
        V = [{}]
        path = {}

        # Initialization
        for tag in self.tags:
            trans_p = self.transition_probs.get(self.start_tag, {}).get(tag, 0)
            emiss_p = self.emission_probs.get(tag, {}).get(sentence[0], 1e-6)
            V[0][tag] = trans_p * emiss_p
            path[tag] = [tag]

        # Recursion
        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for curr_tag in self.tags:
                (prob, state) = max(
                    (V[t - 1][prev_tag] *
                     self.transition_probs.get(prev_tag, {}).get(curr_tag, 0) *
                     self.emission_probs.get(curr_tag, {}).get(sentence[t], 1e-6), prev_tag)
                    for prev_tag in self.tags
                )

                V[t][curr_tag] = prob
                new_path[curr_tag] = path[state] + [curr_tag]

            path = new_path

        # Termination
        n = len(sentence) - 1
        (prob, state) = max(
            (V[n][tag] * self.transition_probs.get(tag, {}).get(self.end_tag, 0), tag)
            for tag in self.tags
        )

        return path[state]


# Training data
tagged_data = [
    [('THE', 'DET'), ('cat', 'NOUN'), ('sleeps', 'VERB')],
    [('A', 'DET'), ('dog', 'NOUN'), ('barks', 'VERB')],
    [('THE', 'DET'), ('dog', 'NOUN'), ('sleeps', 'VERB')],
    [('MY', 'DET'), ('dog', 'NOUN'), ('runs', 'VERB'), ('fast', 'ADV')],
    [('A', 'DET'), ('cat', 'NOUN'), ('meows', 'VERB'), ('loudly', 'ADV')],
    [('YOUR', 'DET'), ('cat', 'NOUN'), ('runs', 'VERB')],
    [('THE', 'DET'), ('bird', 'NOUN'), ('sings', 'VERB'), ('sweetly', 'ADV')],
    [('A', 'DET'), ('bird', 'NOUN'), ('chirps', 'VERB')]
]

model = HMM()
model.train(tagged_data)

# Test data with true tags
test_data = [
    [('The', 'DET'), ('cat', 'NOUN'), ('meows', 'VERB')],
    [('My', 'DET'), ('dog', 'NOUN'), ('barks', 'VERB'), ('loudly', 'ADV')]
]

all_true_tags = []
all_pred_tags = []

for sent in test_data:
    words, true_tags = zip(*sent)
    pred_tags = model.viterbi(words)

    print("Sentence:", words)
    print("True tags:", true_tags)
    print("Predicted tags:", pred_tags)
    print()

    all_true_tags.extend(true_tags)
    all_pred_tags.extend(pred_tags)

accuracy = accuracy_score(all_true_tags, all_pred_tags)
print(f"HMM Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(all_true_tags, all_pred_tags))

