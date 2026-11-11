"""A small predefined recipe search over held-out validation probabilities."""
import numpy as np
from sklearn.metrics import f1_score


def recipe_candidates(probabilities, labels, profiles, qcs_weights=(.25,.5,.75), reference_keys=(),
                      new_keys=(), reference_blend_weights=()):
    candidates=[]
    seen=set()

    def add(name, components):
        signature=tuple(sorted((key,round(weight,12)) for key,weight in components))
        if signature in seen:
            return
        seen.add(signature)
        if not np.isclose(sum(weight for _,weight in components),1.):
            raise ValueError('Recipe weights must sum to one.')
        combined=sum(probabilities[key]*weight for key,weight in components)
        if combined.shape!=(len(labels),6) or not np.isfinite(combined).all():
            raise ValueError('Invalid candidate probabilities.')
        prediction=combined.argmax(1)
        candidates.append(dict(name=name,components=[dict(profile=key,weight=weight) for key,weight in components],
                               accuracy=float(np.mean(prediction==labels)),
                               macro_f1=float(f1_score(labels,prediction,labels=range(6),average='macro',zero_division=0))))

    for key in probabilities:
        add(key,[(key,1.)])
    singles=sorted(candidates,key=lambda r:(r['accuracy'],r['macro_f1']),reverse=True)
    # At most one inference variant of each trained model enters an ensemble.
    ranked=[]
    runs=set()
    for row in singles:
        key=row['components'][0]['profile']
        run=profiles[key]['run']
        same_predictions=any(np.array_equal(probabilities[key],probabilities[previous]) for previous in ranked)
        if run not in runs and not same_predictions:
            ranked.append(key)
            runs.add(run)
    for n in (2,3):
        if len(ranked)>=n:
            add(f'mean_top_{n}',[(key,1./n) for key in ranked[:n]])
    if reference_keys:
        add('reference_mean',[(key,1./len(reference_keys)) for key in reference_keys])
        for key in new_keys:
            if key in reference_keys:
                raise ValueError('New blend candidates must be separate from the reference ensemble.')
            for weight in reference_blend_weights:
                if not 0 < weight < 1:
                    raise ValueError('Reference blend weights must lie strictly between zero and one.')
                add(f'reference_plus_{key}_{weight:.2f}',
                    [(ref,(1-weight)/len(reference_keys)) for ref in reference_keys]+[(key,weight)])
    qcs=next((key for key in ranked if profiles[key]['model']=='qcs'),None)
    other=next((key for key in ranked if profiles[key]['model']!='qcs'),None)
    if qcs is not None and other is not None:
        for weight in qcs_weights:
            add(f'best_non_qcs_plus_qcs_{weight:.2f}',[(other,1-weight),(qcs,weight)])
    return sorted(candidates,key=lambda r:(r['accuracy'],r['macro_f1'],-len(r['components'])),reverse=True)
