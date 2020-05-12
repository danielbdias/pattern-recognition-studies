import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from collections import namedtuple

PCAResult = namedtuple("PCAResult", [ "pca_stats", "principal_components", "principal_components_under_threshold", "X_transformed" ])

def centralize_observations(raw_data):
    """Assumindo a base bruta, arrasta cada observação (frame) para um ponto médio,
       considerando somente as coordenadas x, y."""
    nose_tip_coordinates = ['89x', '89y']

    means = raw_data[nose_tip_coordinates].mean()

    # encontra o delta de cada frame (o quanto o frame precisa se mover para ficar no ponto médio)
    x_delta = raw_data['89x'] - means['89x']
    y_delta = raw_data['89y'] - means['89y']

    # encontra as colunas de cada coordenada
    columns = list(raw_data.columns)
    x_columns = list(filter(lambda column: column.endswith('x'), columns))
    y_columns = list(filter(lambda column: column.endswith('y'), columns))

    # "arrasta" os pontos para o delta encontrado
    normalized_data = raw_data.copy()

    for x_column in x_columns:
        normalized_data[x_column] = abs(normalized_data[x_column] - x_delta)

    for y_column in y_columns:
        normalized_data[y_column] = abs(normalized_data[y_column] - y_delta)

    return normalized_data

def transform_to_distances(data):
    """Assumindo a base com coordenadas (x, y, z), descarta a coordenada z e
       calcula as distâncias entre todos os pontos."""
    euclidean_distance = lambda x0, y0, x1, y1: np.sqrt(np.square(x0 - x1) + np.square(y0 - y1))

    # infere o número de pontos na base identificando as colunas da coordenada x
    total_points = len(list(filter(lambda column: column.endswith('x'), data.columns)))

    data_frame_as_dict = {}
    columns = []

    # variável a variável (coluna a coluna), constroi as novas variáveis de distância
    for first_point_index in list(range(0, total_points - 1)):
        for second_point_index in list(range(first_point_index + 1, total_points)):
            dimension_name = f'p{first_point_index}_p{second_point_index}'

            x0 = data[f'{first_point_index}x']
            y0 = data[f'{first_point_index}y']
            x1 = data[f'{second_point_index}x']
            y1 = data[f'{second_point_index}y']

            data_frame_as_dict[dimension_name] = list(euclidean_distance(x0, y0, x1, y1))
            columns.append(dimension_name)

    data_frame_as_dict['target'] = list(data['target'])
    columns.append('target')

    return pd.DataFrame(data_frame_as_dict, columns = columns)

def principal_component_analysis(features, X, threshold = 80.0):
    """Calcula o PCA com o sklearn"""
    pca = PCA()
    X_transformed = pca.fit_transform(X)

    component_names = list(map(lambda i : f'PC{i}', range(1, len(features) + 1)))

    # como o PCA do sklearn utiliza SVD, ao invés de ele nos retornar os autovalores (eigenvalues),
    # ele nos retorna valores singulares (singular values), que precisam ser convertidos para autovalores
    # referência: https://stackoverflow.com/questions/31909945/obtain-eigen-values-and-vectors-from-sklearn-pca
    eigenvalues = pd.Series(data=(pca.singular_values_ ** 2) / (len(X)), index=component_names)
    variance_ratio = pd.Series(data=pca.explained_variance_ratio_ * 100, index=component_names)

    pca_stats = pd.DataFrame(data={
        'Eigenvalues': eigenvalues,
        'Variance %': variance_ratio,
        'Cum. Eigenvalues': eigenvalues.cumsum(),
        'Cum. Variance %': variance_ratio.cumsum()
    })

    principal_components = pd.DataFrame(data=pca.components_.T, columns=component_names, index=features)

    # encontra o PC que faz ultrapassa o limiar
    threshold_expression = pca_stats['Cum. Variance %'] >= threshold
    threshold_index = pca_stats[threshold_expression].head(1).index.values

    # mostra todos os pcs dentro do limiar mais o primeiro que passou o limiar
    principal_components_under_threshold = list(pca_stats[~threshold_expression].index.values) + list(threshold_index)

    return PCAResult(pca_stats, principal_components, principal_components_under_threshold, X_transformed)
