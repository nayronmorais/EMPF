"""That package provide implementation of methods monitoring of industrial process."""


class Monitor:
    """Monitoring class.

    Use this class for monitoring a process.

    Parameters
    ----------
    clustering_method : tuple<method, config>
        Reference for clustering method and your configuration.

    diagnosis_method : tuple<method, config>
        Reference for diagnosis method and your configuration.

    Attributes
    ----------
    clus_method : class of clustering_method<method>
        Clustering method.

    diag_method : class of diagnosis_method<method>
        Diagnosis method.

    """

    def __init__(self, clustering_method, diagnosis_method):

        clus_method, init_config_clus = clustering_method
        diag_method, init_config_diag = diagnosis_method

        self.clus_method = clus_method(**init_config_clus)
        self.diag_method = diag_method(**init_config_diag)

    def process_input(self, x):
        """
        Process a input and return the assigment values.

        Parameters
        ----------
        x : numpy.ndarray, shape=(1, n)
            New sample.

        Returns
        -------
        isknow : bool
            If the sample it's in cluster transition.

        cluster_assig : int
            Assigment cluster for input.

        diag : the format of diag_method.process_input
            Diagnosis for the input.

        """
        isknow, cluster_assig = self.clus_method.process_input(x)

        diag = self.diag_method.process_input(x, cluster_assig, isknow)

        return isknow, cluster_assig, diag
