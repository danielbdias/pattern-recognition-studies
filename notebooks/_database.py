import pandas as pd

categories = ['wh_question', 'yn_question', 'doubt_question', 'negative', 'affirmative', 'conditional', 
              'relative', 'topics', 'emphasis']

def load_datapoints(category):    
    a = pd.read_table(f'../database/a_{category}_datapoints.txt', sep = ' ')
    b = pd.read_table(f'../database/b_{category}_datapoints.txt', sep = ' ')
    
    data = pd.concat([a, b])
    return data.drop(['0.0'], axis=1) # ignora a primeira coluna (timespamp do frame)

def load_targets(category):        
    a = pd.read_table(f'../database/a_{category}_targets.txt', sep = ' ', header = None, names = ['target'])
    b = pd.read_table(f'../database/b_{category}_targets.txt', sep = ' ', header = None, names = ['target'])
    
    return pd.concat([a, b])

def load_datapoints_with_targets(category):
    datapoints = load_datapoints(category)
    targets = load_targets(category)
    
    return datapoints.assign(target = targets['target'])

def single_category_database_instance_description(category):
    datapoints = load_datapoints_with_targets(category)
    positive_datapoints = datapoints[datapoints['target'] == 1]
    negative_datapoints = datapoints[datapoints['target'] == 0]
    
    total_instances = len(datapoints.index)
    total_instances_classified_as_positive = len(positive_datapoints.index)
    total_instances_classified_as_negative = len(negative_datapoints.index)
    
    return pd.DataFrame(data={
        'Positive Instances': [total_instances_classified_as_positive],
        'Positive Instance Proportion': [ float(total_instances_classified_as_positive) / total_instances ],
        'Negative Instances': [total_instances_classified_as_negative],
        'Negative Instance Proportion': [ float(total_instances_classified_as_negative) / total_instances ],
        'Total Instances': [total_instances]
    })

def all_categories_database_instance_description():
    descriptions = map(lambda category : single_category_database_instance_description(category), categories)
    
    instance_description = pd.concat(descriptions)
    instance_description.insert(0, "Facial Expression", categories)
    return instance_description.reset_index(drop = True)