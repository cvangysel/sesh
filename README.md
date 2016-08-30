Sesh
----

*Sesh* is a testbed for the session search task in information retrieval. In particular, it allows the evaluation of various session-aware retrieval algorithms on the [TREC Session Track](http://trec.nist.gov/data/session.html). It was released as part of the [ICTIR 2016](https://arxiv.org/abs/1608.06656) paper on lexical query modeling in session search.

Prerequisites
-------------

Sesh requires Python 3.5, [Indri 5.9](http://www.lemurproject.org/indri/) with [Pyndri](https://github.com/cvangysel/pyndri), [trec_eval](https://github.com/usnistgov/trec_eval) and assorted [modules](requirements.txt).

Usage
-----

To replicate the experiments of the paper on lexical query modeling in session search you will need Indri indexes and [link files](http://lemur.sourceforge.net/indri/HarvestLinks.html) of the ClueWeb09b and ClueWeb12b. We cannot help you with this as we do not have redistribution rights.

__NOTE__: Make sure that Indri 5.9 has been correctly installed before attempting the tutorial below. That means that Indri's executables are in your `PATH`, its headers are in your `CPATH` and its library directories are in your `LD_LIBRARY_PATH`.

To begin, create a virtual Python environment and install dependencies:

    [cvangysel@ilps cvangysel] git clone git@github.com:cvangysel/sesh.git
    [cvangysel@ilps cvangysel] cd sesh

    [cvangysel@ilps sesh] virtualenv sesh-dev
    Using base prefix '/Users/cvangysel/anaconda3'
    New python executable in /home/cvangysel/sesh/sesh-dev/bin/python
    Installing setuptools, pip, wheel...done.

    [cvangysel@ilps sesh] source sesh-dev/bin/activate

    (sesh-dev) [cvangysel@ilps sesh] pip install numpy==1.11.1  # First numpy, as it is needed during installation of other packages.
    (sesh-dev) [cvangysel@ilps sesh] pip install -r requirements.txt

    (sesh-dev) [cvangysel@ilps sesh] git clone git@github.com:cvangysel/pyndri.git

    # This step will fail if Indri is not correctly installed. If so, install Indri first!
    (sesh-dev) [cvangysel@ilps sesh] cd pyndri && python setup.py install && python tests/pyndri_tests.py

Once setup has finished, we can download the TREC Session Track logs. All files will be downloaded to the `scratch` subdirectory of your current working directory.

    (sesh-dev) [cvangysel@ilps sesh] scripts/fetch_session_logs.sh
    Fetching 2011 logs: http://trec.nist.gov/data/session/11/sessiontrack2011.RL4.xml
    Fetching 2011 judgments: http://trec.nist.gov/data/session/11/judgments.txt
    Fetching 2011 session-topic map: http://trec.nist.gov/data/session/11/sessionlastquery_subtopic_map.txt
    Creating 2011 trec_eval-compatible qrel.
    Fetching 2013 logs: http://trec.nist.gov/data/session/2013/sessiontrack2013.xml
    Fetching 2013 judgments: http://trec.nist.gov/data/session/2013/qrels.txt
    Fetching 2013 session-topic map: http://trec.nist.gov/data/session/2013/sessiontopicmap.txt
    Creating 2013 trec_eval-compatible qrel.
    Fetching 2012 logs: http://trec.nist.gov/data/session/12/sessiontrack2012.txt
    Fetching 2012 judgments: http://trec.nist.gov/data/session/12/qrels.txt
    Fetching 2012 session-topic map: http://trec.nist.gov/data/session/12/sessiontopicmap.txt
    Creating 2012 trec_eval-compatible qrel.
    Fetching 2014 logs: http://trec.nist.gov/data/session/2014/sessiontrack2014.xml
    Fetching 2014 judgments: http://trec.nist.gov/data/session/2014/judgments.txt
    Fetching 2014 session-topic map: http://trec.nist.gov/data/session/2014/session-topic-mapping.txt
    Creating 2014 trec_eval-compatible qrel.

Finally, you can run experiment configurations. We have packaged configuration files for the ICTIR paper experiments in the `configs` subdirectory:

   * [configs/all.pb](configs/all.pb): Table 2, Figure 1 and Figure 2
   * [configs/progressing_session.pb](configs/progressing_session.pb): Figure 3a
   * [configs/markov_context.pb](configs/markov_context.pb): Figure 3b
   * [configs/brute.pb](configs/brute.pb): Table 3 (however, __fair warning__: in the paper we used a proprietary implementation that parallelized the computation on a compute cluster. In this OSS version, brute-forcing will run on a single core and may take a very long time (i.e., weeks to months) to finish.)

For example, we can run the main experiment on the 2011 TREC Session Track as follows:

    (sesh-dev) [cvangysel@ilps sesh] scripts/score_session_log.sh \
        2011 configs/all.pb \
        <PATH TO CLUEWEB09B INDRI INDEX> \
        <PATH TO CLUEWEB09B HARVESTLINKS OUTPUT>

    Evaluating scratch/2011/2011.qrel for scratch/output/all.pb/2011/indri_all.run.
    ndcg_cut_10            all     0.4476
    Evaluating scratch/2011/2011.qrel for scratch/output/all.pb/2011/oracle.run.
    ndcg_cut_10            all     0.7771
    Evaluating scratch/2011/2011.qrel for scratch/output/all.pb/2011/qcm.run.
    ndcg_cut_10            all     0.4397
    Evaluating scratch/2011/2011.qrel for scratch/output/all.pb/2011/indri_first.run.
    ndcg_cut_10            all     0.3712
    Evaluating scratch/2011/2011.qrel for scratch/output/all.pb/2011/nugget_RL4.run.
    ndcg_cut_10            all     0.4367
    Evaluating scratch/2011/2011.qrel for scratch/output/all.pb/2011/nugget_RL3.run.
    ndcg_cut_10            all     0.4420
    Evaluating scratch/2011/2011.qrel for scratch/output/all.pb/2011/indri_last.run.
    ndcg_cut_10            all     0.3576
    Evaluating scratch/2011/2011.qrel for scratch/output/all.pb/2011/nugget_RL2.run.
    ndcg_cut_10             all     0.4367

All the experiment output (debug files, execution logs, runs, trec_eval output) can be found in `scratch/output/all.pb/2011`.

Citation
--------

If you use Sesh to produce results for your scientific publication, please refer to [this paper](https://arxiv.org/pdf/1608.06656v1.pdf):

```
@inproceedings{VanGysel2016-ICTIR,
  title={Lexical Query Modeling in Session Search},
  author={Van Gysel, Christophe and Kanoulas, Evangelos and de Rijke, Maarten},
  booktitle={ICTIR},
  volume={2016},
  year={2016},
  organization={ACM}
}
```

License
-------

Sesh is licensed under the [MIT license](LICENSE). If you modify Sesh in any way, please link back to this repository.
