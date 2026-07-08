def apply_behavioural_mapping(action, avg_portfolio_return, data,date,config): #forecast has the equal weighted portfolio return
    #new weights
    action = np.asarray(action)
    n = 1
    if (avg_portfolio_return > config.THRESHOLD_PARAMETER):
        action = _apply_overconfidence(action,data,date,config)
        n = config.oc_n
    elif (avg_portfolio_return < -config.THRESHOLD_PARAMETER):
        action = _apply_loss_averse(action,data,date,config)
        n = config.ra_n
    #normalize the action values to ensure they sum to 1
    weight_sum = action.sum()
    if weight_sum != 0:
        action = action/weight_sum
    action = action*n   
    return action

def _apply_overconfidence(action,data,date,config):
    for i in range(len(action)):
        m=1
        if(data[data['unique_id']==config.ticker_list[i]][data['ds']==date]['y'].item() >= config.oc_upper_threshold):
            m = 1 + config.oc_k_gain
        if(data[data['unique_id']==config.ticker_list[i]][data['ds']==date]['y'].item() <= config.oc_lower_threshold):
            m = 1 + config.oc_k_loss        
        action[i] *= m
    return action

def _apply_loss_averse(action,data,date,config):
    for i in range(len(action)):
        m=1
        if(data[data['unique_id']==config.ticker_list[i]][data['ds']==date]['y'].item() >= config.ra_upper_threshold):
            m = 1 + config.ra_k_gain
        if(data[data['unique_id']==config.ticker_list[i]][data['ds']==date]['y'].item() <= config.ra_lower_threshold):
            m = 1 + config.ra_k_loss        
        action[i] *= m
    return action