from amber.architect import reward

def deserilizer(s):
    if s == 'AUC':
        return reward.AucReward(method='auc')
    elif s == 'F1':
        return reward.F1Reward()
    else:
        raise Exception("unknown string id: %s" %s)