
class Client(object):

    def __init__(self, hostId, clientId, speed, traces=None):
        self.hostId = hostId
        self.clientId = clientId
        #Ahmed - based on the format from the device trace file key:432802 val:{'computation': 162.0, 'communication': 5648.109619499134}
        self.compute_speed = speed['computation']
        self.bandwidth = speed['communication']
        self.score = 0
        self.traces = traces
        self.behavior_index = 0

    def getScore(self):
        return self.score

    def registerReward(self, reward):
        self.score = reward

    #TODO: clarify this part on the use of the trace!
    #Ahmed - the trace pickle file contains only 107,749 clients!
    #Format- key:3834 val:{'duration': 211625, 'inactive': [65881, 133574, 208292, 276575, 295006, 356236, 400906, 475099], 'finish_time': 518400, 'active': [12788, 100044, 188992, 271372, 276663, 352625, 356267, 441193], 'model': 'CPH1801'}
    def isActive(self, cur_time):
        if self.traces is None:
            return True

        norm_time = cur_time % self.traces['finish_time']

        if norm_time > self.traces['inactive'][self.behavior_index]:
            self.behavior_index += 1

        self.behavior_index %= len(self.traces['active'])

        if (self.traces['active'][self.behavior_index] <= norm_time <= self.traces['inactive'][self.behavior_index]):
            return True
        return False

    #Ahmed - return the availability windows of the client
    def availabilityPeriods(self):
        period_list=[]
        for i in range(len(self.traces['inactive'])):
            period_list.append((self.traces['active'][i], self.traces['inactive'][i]))
        return period_list

    #TODO clarify the contents of the device compute trace
    #Ahmed - the trace pickle file contains only 500,000 clients!
    #Format - key:432802 val:{'computation': 162.0, 'communication': 5648.109619499134}
    def getCompletionTime(self, batch_size, upload_epoch, upload_size, download_size, augmentation_factor=3.0):
        """
           Computation latency: compute_speed is the inference latency of models (ms/sample). As reproted in many papers, 
                                backward-pass takes around 2x the latency, so we multiple it by 3x;
           Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        """
        return {'computation':augmentation_factor * batch_size * upload_epoch*float(self.compute_speed)/1000., \
                'communication': (upload_size+download_size)/float(self.bandwidth)}
