import os

cdef bytes as_str(data):
    if isinstance(data, bytes):
        return data
    elif isinstance(data, unicode):
        return data.encode('utf8')
    raise TypeError('Cannot convert %s to string' % type(data))

cdef class LanguageModel:
    cdef Model* model
    cdef public bytes path
    cdef const_Vocabulary* vocab
    cdef unsigned startsym
    cdef unsigned stopsym
    cdef unsigned unk

    def __init__(self, path):
        self.path = os.path.abspath(as_str(path))
        try:
            self.model = LoadVirtual(self.path)
        except RuntimeError as exception:
            exception_message = str(exception).replace('\n', ' ')
            raise IOError('Cannot read model \'{}\' ({})'.format(path, exception_message))\
                    from exception
        self.vocab = &self.model.BaseVocabulary()
        cdef unsigned startsym, stopsym, unk
        self.startsym = self.vocab.BeginSentence()
        self.stopsym = self.vocab.EndSentence()
        self.unk = self.vocab.NotFound()

    def __dealloc__(self):
        del self.model

    property order:
        def __get__(self):
            return self.model.Order()
    
    def score(self, sentence):
        cdef list words = as_str(sentence).split()
        cdef State state
        self.model.BeginSentenceWrite(&state)
        cdef State out_state
        cdef double total = 0
        for word in words:
            total += self.model.BaseScore(&state, self.vocab.Index(word), &out_state)
            state = out_state
        total += self.model.BaseScore(&state, self.vocab.EndSentence(), &out_state)
        return total

    def full_scores(self, sentence):
        cdef list words = as_str(sentence).split()
        cdef State state
        self.model.BeginSentenceWrite(&state)
        cdef State out_state
        cdef FullScoreReturn ret
        cdef double total = 0
        for word in words:
            ret = self.model.BaseFullScore(&state,
                self.vocab.Index(word), &out_state)
            yield (ret.prob, ret.ngram_length)
            state = out_state
        ret = self.model.BaseFullScore(&state,
            self.vocab.EndSentence(), &out_state)
        yield (ret.prob, ret.ngram_length)
            
    def __contains__(self, word):
        cdef bytes w = as_str(word)
        return (self.vocab.Index(w) != 0)

    def __repr__(self):
        return '<LanguageModel from {0}>'.format(os.path.basename(self.path))

    def __reduce__(self):
        return (LanguageModel, (self.path,))
    
    def to_index(self, word_tuple):
        ret = []
        cdef int id
        for word in word_tuple:
            id = self.vocab.Index(word)
            ret.append(id)
        return ret
    
    property start_symbol_index:
        def __get__(self):
            return self.startsym
        
    property stop_symbol_index:
        def __get__(self):
            return self.stopsym
        
    property unk_symbol_index:
        def __get__(self):
            return self.unksym
        
    def check_start_middle_stop(self, context):
        if len(context) <= 0:
            return context
        ret = []
        sent_start = False
        sent_stop = False
        start = 0
        stop = len(context)
        if context[0] == self.startsym:
            sent_start = True
            start += 1
        if context[-1] == self.stopsym:
            sent_stop = True
            stop -= 1
        return sent_start, context[start:stop], sent_stop
    
    def pscore(self, context):
        cdef double total = 0
        cdef State state
        cdef unsigned word_index
        sent_start, context, sent_stop = self.check_start_middle_stop(context)
        if sent_start:
            self.model.BeginSentenceWrite(&state)
        else:
            self.model.NullContextWrite(&state)
        cdef State out_state
        length = len(context) if len(context) < (self.order - 1) else (self.order - 1) #only calculate the first part
        for index in context[:length]:
            word_index = index
            total += self.model.BaseScore(&state, word_index, &out_state)
            state = out_state
        #ignore sent stop here, you will get it on the qscore part
        return total
    
    def qscore(self, context):
        cdef double total = 0
        if len(context) < self.order:
            return total
        cdef State state
        cdef unsigned word_index
        cdef double cur_score
        sent_start, context, sent_stop = self.check_start_middle_stop(context)
        if sent_start:
            self.model.BeginSentenceWrite(&state)
        else:
            self.model.NullContextWrite(&state)
        cdef State out_state
        for i, index in enumerate(context):
            word_index = index
            cur_score = self.model.BaseScore(&state, word_index, &out_state)
            if i >= (self.order - 1):
                total += cur_score
            state = out_state
        if sent_stop:
            total += self.model.BaseScore(&state, self.vocab.EndSentence(), &out_state)
            state = out_state
        return total  #end without end sym, which actually wil be put in as </s> (2)
    
    def pqscore(self, context):
        cdef double total = 0
        cdef State state
        cdef unsigned word_index
        sent_start, context, sent_stop = self.check_start_middle_stop(context)
        if sent_start:
            self.model.BeginSentenceWrite(&state)
        else:
            self.model.NullContextWrite(&state)
        cdef State out_state
        for index in context:
            word_index = index
            total += self.model.BaseScore(&state, word_index, &out_state)
            state = out_state
        if sent_stop:
            total += self.model.BaseScore(&state, self.vocab.EndSentence(), &out_state)
            state = out_state
        return total  #end without end sym, which actually wil be put in as </s> (2)
    

