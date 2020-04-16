import pandas as pd
import numpy as np

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
    total_points = len(list(filter(lambda column: column.endswith('x'), raw_data.columns)))

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

