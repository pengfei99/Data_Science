
def remean_points(mean_points, *args):
    points=args[0]
    return points-mean_points


def name_index(df):
    df.index.name = 'review_id'
    return df