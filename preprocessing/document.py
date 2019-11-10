from collections import defaultdict
import re
import numpy as np


class Document:
    def __init__(self, doc, annotations):
        self._doc = doc
        self._annotations = annotations

    @classmethod
    def from_file(cls, doc_path, nlp):
        with open(doc_path) as f:
            doc = f.read()
        
        #### remember all annotations and remove them from document ####
        shift = 0
        doc_annotations = defaultdict(list)
        all_matched_people = list(re.finditer(r'<Entity name=".*?" type="person" category=".*?">.*?</Entity>', doc))
        for match_obj in all_matched_people:
            entity_match, entity_start, entity_end = match_obj.group(0), match_obj.start(0) - shift, match_obj.end(0) - shift
            person = entity_match[14:entity_match.find('type="person"') - 2]
            category_match = re.search(r'category=".*?">', entity_match)
            category = category_match.group(0)[10:-2]
            name_start, name_end = entity_start + category_match.end(0), entity_end - 9
            name_parts = len(doc[name_start:name_end].split(' '))
            doc = doc[:entity_start] + doc[name_start:name_end] + doc[entity_end:]
            shift += entity_end - name_end + name_start - entity_start
            doc_annotations[(person, category)].append((entity_start, name_parts))  # entity_start + name_end - name_start))

        #### tokenize text ####
        parsed_doc = nlp(doc)
        words_positions = np.array([token.idx for token in parsed_doc if not token.is_punct or token.text in ['.', '!', '?']])
        words = [token.text for token in parsed_doc if not token.is_punct or token.text in ['.', '!', '?']]

        for (person, category), annotations in doc_annotations.items():
            doc_indices = [(np.argwhere(words_positions==name_pos), name_parts) for name_pos, name_parts in annotations]
            doc_indices = [(name_word_pos[0][0], name_parts) for name_word_pos, name_parts in doc_indices if name_word_pos.size > 0]
            doc_annotations[(person, category)] = doc_indices

        return cls(words, doc_annotations)

    def get_neighbors(self, neighborhood):
        if isinstance(neighborhood, int) and neighborhood > 0:
            return self._window_neighbourhood(neighborhood)
        elif neighborhood == 'sentence':
            return self._sentence_neighborhood()
        raise RuntimeError(f'Unknown method: {neighborhood}. Should be positive integer or "sentence"')

    def _window_neighbourhood(self, window_size):
        neighbors = defaultdict(list)
        for (person, category), annotations in self._annotations.items():
            for name_position, name_parts in annotations:
                start, end = max(0, name_position - window_size), min(len(self._doc), name_position + name_parts + window_size)
                neighbors[(person, category)].append(self._doc[start:name_position] + self._doc[name_position+name_parts:end])
        return neighbors
                
    def _sentence_neighborhood(self):
        neighbors = defaultdict(list)
        stop_tokens = ['.', '!', '?']
        for (person, category), annotations in self._annotations.items():
            for name_position, name_parts in annotations:
                start, end = name_position, name_position + name_parts
                while start > 0 and self._doc[start] not in stop_tokens:
                    start -= 1
                if start != 0: start += 1

                while end < len(self._doc) and self._doc[end] not in stop_tokens:
                    end += 1

                neighborhood = self._doc[start:name_position] + self._doc[name_position+name_parts:end]
                if neighborhood:
                    neighbors[(person, category)].append(neighborhood)
        return neighbors
