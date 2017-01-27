# xgboost jvm and xgboost python return different metric results

How can it be that the python version and xgboost4j return different metrics? E.g. for a kappa these metrics will differ by quite a lot. 


You should see something similar to the following in spark
```
MeasureUnit(kappa,0.4086460032626426)
MeasureUnit(f1_R,0.7563025210084033)
MeasureUnit(AUC_R,0.7011240465676435)
```
Here Metrics for kappa are around 0.3 up to 0.8 where in python these are strictly 1 (over-fit)

As you can see there is quite some difference between the results of xgboost in python and in spark. Depending of the specific values, the difference between what python and what xgboost in spark report on my real data-set are $|(metric_{python} -metric_{spark}|$ up to $0.3$ apart What is wrong here?

**Looking forward for any hints.**
