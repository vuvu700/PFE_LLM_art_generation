from collections import defaultdict

class Historique():
    
    def __init__(self):
        self.informations:dict[str, dict[int, float |int]] = {}
        self.commentaries:dict[int, list[str]] = defaultdict(list)

    #ajout metrics et commentaires
    def add_metric(self, metric_name: str, metric_value: float | int, epoch_id: int):
        if metric_name not in self.informations:
            self.informations[metric_name] = {}
        self.informations[metric_name][epoch_id] = metric_value

    def add_commentaries(self, epoch_id, commentarie: str):
        self.commentaries[epoch_id].append(commentarie)

    #voir metrics
    def get_metric_value(self, metric_name: str, epoch_id:int):
        return self.informations.get(metric_name, {}).get(epoch_id)
    
    def get_all_historique_of_one_metric(self, metric_name:str):
        return self.informations.get(metric_name, {})
    
    def get_all_historique(self):
        return self.informations
    
    def get_all_metrics_name(self):
        return list(self.informations.keys())
    

    # voir commentaires
    def get_commentaries_value(self,  epoch_id:int):
        return self.commentaries.get(epoch_id)
    
    def get_all_commentaries(self):
        return self.commentaries

    
    
    #def load_json(self):

    #def save_json(self):

