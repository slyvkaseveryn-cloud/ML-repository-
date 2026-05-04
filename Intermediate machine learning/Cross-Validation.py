def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.

    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
    ])

    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')

    return scores.mean()

# Check your answer
step_1.check()


results = {}
for i in range(1, 9):
    results[50*i] = get_score(50*i)

n_estimators_best = min(results, key=results.get)

step_2.check()

n_estimators_best = min(results, key=results.get)

# Check your answer
step_3.check()
