from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def process(self, words_list: 'typing.List[typing.List[str]]') -> 'typing.List[np.ndarray]':
        """ 
        Processes list of sentences (collections of words) converting each word into its 
        embedding and aggregates them into one vector per sentence 

        params: 
        words_list -- List of lists of words. 

        returns:
        List of embeddings (np.ndarray) per sentence. Each embedding has one dimension of the same size.
        """
        pass
