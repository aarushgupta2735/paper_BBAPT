
def apply_behavioural_mapping(action, forecast, data,date,config): #forecast has the equal weighted portfolio return
    #new weights
    n = 1
    if (forecast['avg_return'].iloc[0] > config.THRESHOLD_PARAMETER):
        for ticker in config.ticker_list:
            action = _apply_overconfidence(action, ticker,data,date,config)
        n = config.oc_n
    elif (forecast['avg_return'].iloc[0] < -config.THRESHOLD_PARAMETER):
        for ticker in config.ticker_list:
            action = _apply_loss_averse(action, ticker,data,date,config)
        n = config.ra_n
    #normalize the action values to ensure they sum to 1
    weight_sum = sum(action.values())
    if weight_sum != 0:
        action = {ticker: action[ticker]/weight_sum for ticker in config.ticker_list}
    action = {ticker: action[ticker]**n for ticker in config.ticker_list}
    return action

def _apply_overconfidence(action, ticker,data,date,config):
    m = 1
    if(data[data['unique_id']==ticker][data['ds']==date]['y'] >= config.oc_upper_threshold):
        m = 1 + config.oc_k_gain
    if(data[data['unique_id']==ticker][data['ds']==date]['y'] <= config.oc_lower_threshold):
        m = 1 + config.oc_k_loss        
    action[ticker] *= m
    return action

def _apply_loss_averse(action, ticker,data,date,config):
    m = 1
    if(data[data['unique_id']==ticker][data['ds']==date]['y'] >= config.ra_upper_threshold):
        m = 1 + config.ra_k_gain
    if(data[data['unique_id']==ticker][data['ds']==date]['y'] <= config.ra_lower_threshold):
        m = 1 + config.ra_k_loss        
    action[ticker] *= m
    return action