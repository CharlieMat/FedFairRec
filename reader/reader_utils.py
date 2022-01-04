import numpy as np

############################################################################
#                          Raw Data Transformer                            #
############################################################################

def v2id(features, vocab, field_name):
    '''
    field value to single id in vocab
    @input:
    - features: {field_name: value}
    - vocab: {(field_name, value): {'idx': idx}}
    - field_name
    @output:
    - idx
    '''
    if field_name not in features or features[field_name] not in vocab[field_name]:
        return 0
    else:
        return vocab[field_name][features[field_name]]
    
def v2multid(features, vocab, field_name, max_n_value = 15):
    '''
    field value to multiple ids in vocab
    @input:
    - features: {field_name: 'value'}
    - vocab: {(field_name, value): {'idx': idx}}
    - field_name
    - max_n_value: max number of values to include for the field
    @output:
    - idx
    '''
    vec = []
    try:
        if field_name in features and len(features[field_name]) > 0:
            for v in features[field_name].split(","):
                if v in vocab[field_name]:
                    vec.append(vocab[field_name][v])
                else:
                    vec.append(0)
    except:
        return np.array([0] * max_n_value)
    if len(vec) < max_n_value:
        vec = [0] * (max_n_value - len(vec)) + vec
    return np.array(vec[-max_n_value:])

def v2onehot(features, vocab, field_name):
    '''
    field values to one-hot or multi-hot vector
    @input:
    - features: {field_name: 'value'}
    - vocab: {(field_name, value): {'idx': idx}}
    - field_name
    @output:
    - idx
    '''
    vec = [0.] * (len(vocab[field_name])+1)
#     if field_name in features and len(features[field_name]) > 0:
    try:
        for v in features[field_name].split(","):
            if v in vocab[field_name]:
                vec[vocab[field_name][v]] = 1.
    except:
        vec[0] = 1.
    return np.array(vec)

def padding_and_cut(sequence, max_len):
    if len(sequence) < max_len:
        return [0] * (max_len - len(sequence)) + sequence
    else:
        return sequence[-max_len:]

# def nominal_transfer_map(meta_data, col_name):
#     '''
#     @input:
#     - meta_data: pd.dataFrame
#     - col_name: name of the column
    
#     @output:
#     - vMap: (minId, maxId)
#     - f: mapping function that returns the mapped_value given original_value
#     '''
#     data = meta_data[meta_data[col_name].notnull()]
#     vMap = {"NaN": 0}
#     nextId = 1
#     for v in data[col_name].unique():
#         if v not in vMap:
#             vMap[v] = nextId
#             nextId += 1
#     return (0,nextId-1), lambda x: vMap[x] if x in vMap else 0

# def continuous_transfer_map(meta_data, col_name, low = 0.0, high = 1.0):
#     '''
#     @input:
#     - meta_data: pd.dataFrame
#     - col_name: name of the column
#     - low, high: the output range [low, high]
    
#     @output:
#     - range: (minimum value, maximum value)
#     - f: mapping function that returns the mapped_value given original_value
#     '''
#     data = meta_data[meta_data[col_name].notnull()]
#     minV, maxV = data[col_name].min(), data[col_name].max()
#     epsilon = 1e-4
#     logMin, logMax = np.log(1e-4), np.log(maxV - minV + epsilon)
#     return (minV, maxV), lambda x: low + (np.log(x - minV + epsilon) - logMin) * (high - low) / (logMax - logMin)

# def nominal_sequence_transfer_map(meta_data, col_name, separator = " "):
#     '''
#     @input:
#     - meta_data: pd.dataFrame
#     - col_name: name of the column
#     - separator: string separator
    
#     @output:
#     - vMap: (min ID, max ID)
#     - f: mapping function that returns the mapped_sequence given original_sequence
#     '''
#     data = meta_data[meta_data[col_name].notnull()]
#     vMap = {"NaN": 0}
#     nextId = 1
#     for row in data[col_name].str.split(separator):
#         for v in row:
#             if v not in vMap:
#                 vMap[v] = nextId
#                 nextId += 1
#     return (0,nextId-1), (lambda x: [vMap[v] for v in x.split(separator) if v in vMap])



#############################################################################
#                        Data Specific Transformer                          #
#############################################################################

# def transform_ml1m_user(user_meta):
#     '''
#     @input:
#     - user_meta: col_name = [Gender, Age, Occupation, Zip-code]
    
#     @output:
#     - {col_name: value map}
#     - {col_name: mapping function}
#     '''
#     genderMap, genderFunc = nominal_transfer_map(user_meta, "Gender")
#     ageMap, ageFunc = nominal_transfer_map(user_meta, "Age")
#     occMap, occFunc = nominal_transfer_map(user_meta, "Occupation")
#     return {"Gender": genderMap, "Age": ageMap, "Occupation": occMap}, {"Gender": genderFunc, "Age": ageFunc, "Occupation": occFunc}
    
# def transform_bx_user(user_meta):
#     '''
#     @input:
#     - user_meta: col_name = [Location, Age]
    
#     @output:
#     - {col_name: value map}
#     - {col_name: mapping function}
#     '''
#     locMap, locFunc = nominal_sequence_transfer_map(user_meta, "Location", separator = ",")
#     ageMap, ageFunc = continuous_transfer_map(user_meta, "Age", low = 0.0, high = 1.0)
#     return {"Location": locMap, "Age": ageMap}, {"Location": locFunc, "Age": ageFunc}


# def transform_ml1m_item(item_meta):
#     '''
#     @input:
#     - user_meta: col_name = [Title, Genre]
    
#     @output:
#     - {col_name: value map}
#     - {col_name: mapping function}
#     '''
#     genreMap, genreFunc = nominal_sequence_transfer_map(item_meta, "Genre", separator = "|")
#     return {"Genre": genreMap}, {"Genre": genreFunc}


# def transform_ml10m_item(item_meta):
#     '''
#     @input:
#     - user_meta: col_name = [Title, Genre]
    
#     @output:
#     - {col_name: value map}
#     - {col_name: mapping function}
#     '''
#     genreMap, genreFunc = nominal_sequence_transfer_map(item_meta, "Genres", separator = "|")
#     return {"Genres": genreMap}, {"Genres": genreFunc}


# def transform_ml20m_item(item_meta):
#     '''
#     @input:
#     - user_meta: col_name = [Title, Genre]
    
#     @output:
#     - {col_name: value map}
#     - {col_name: mapping function}
#     '''
#     genreMap, genreFunc = nominal_sequence_transfer_map(item_meta, "Genres", separator = "|")
#     return {"Genres": genreMap}, {"Genres": genreFunc}


# def transform_bx_item(item_meta):
#     '''
#     @input:
#     - user_meta: col_name = [Book-Title, Book-Author, Year-Of-Publication, Publisher, ...]
    
#     @output:
#     - {col_name: value map}
#     - {col_name: mapping function}
#     '''
#     authorMap, authorFunc = nominal_transfer_map(item_meta, "Book-Author")
#     yearMap, yearFunc = nominal_transfer_map(item_meta, "Year-Of-Publication")
#     pubMap, pubFunc = nominal_transfer_map(item_meta, "Publisher")
#     return {"Book-Author": authorMap, "Year-Of-Publication": yearMap, "Publisher": pubMap}, \
#             {"Book-Author": authorFunc, "Year-Of-Publication": yearFunc, "Publisher": pubFunc}