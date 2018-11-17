'''the basic class of an RS'''
class Recommender_Base(object):
    def __init__(self):
        self.config = None
        self.dao = None
        self.name = None
        self.type = None

    def showRecommenderInfo(self):
        if self.config is None:
            print('current recommender has not been instantiated!')
        else:
            print('Algorithm name: '+self.name)
            print('Algorithm type: '+self.type)
            self.config.showConfig()

    def Training(self):
        pass

    def Testing(self):
        pass

