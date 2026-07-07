def apply_behavioural_mapping(self,action, forecast, threshold): #forecast has the equal weighted portfolio return
    #new weights
    n = 1
    if (forecast['avg_return'].iloc[0] > threshold):
        for ticker in self.ticker_list:
            action = self._apply_overconfidence(action, ticker)
        n = self.oc_n
    elif (forecast['avg_return'].iloc[0] < -threshold):
        for ticker in self.ticker_list:
            action = self._apply_loss_averse(action, ticker)
        n = self.ra_n
    #normalize the action values to ensure they sum to 1
    weight_sum = sum(action.values())
    if weight_sum != 0:
        action = {ticker: action[ticker]/weight_sum for ticker in self.ticker_list}
    action = {ticker: action[ticker]**n for ticker in self.ticker_list}
    return action

def _apply_overconfidence(self, action, ticker):
    #increase the action value for the given ticker by 10%
    m = 1
    if(self.data[self.data['unique_id']==ticker][self.data['ds']==self.date]['y'] >= self.oc_upper_threshold):
        m = 1 + self.oc_k_gain
    if(self.data[self.data['unique_id']==ticker][self.data['ds']==self.date]['y'] <= self.oc_lower_threshold):
        m = 1 + self.oc_k_loss        
    action[ticker] *= m
    return action

def _apply_loss_averse(self, action, ticker):
    #decrease the action value for the given ticker by 10%
    m = 1
    if(self.data[self.data['unique_id']==ticker][self.data['ds']==self.date]['y'] >= self.ra_upper_threshold):
        m = 1 + self.ra_k_gain
    if(self.data[self.data['unique_id']==ticker][self.data['ds']==self.date]['y'] <= self.ra_lower_threshold):
        m = 1 + self.ra_k_loss        
    action[ticker] *= m
    return action