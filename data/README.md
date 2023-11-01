* fiducial omegam=0.307115
* redshift range:
  * LRG: 0.6~1.0
  * ELG: 0.6~1.1
  * X: 0.6~1.0
* effective redshift of data:
  * LRG:
    * NGC: 0.696
    * SGC: 0.705
    * NGC+SGC: 0.699
  * ELG:
    * NGC: 0.849
    * SGC: 0.841
    * NGC+SGC: 0.845
  * X:
    * NGC: 0.763
    * SGC: 0.774
    * NGC+SGC: 0.770
  * effective redshifts are computed using eq(5) in https://arxiv.org/abs/2007.09010
* `data` folder contains:
  * noric: EZmock pk w/o systematics
  * ric: EZmock pk w/o systematics, w/ radial integral constraint (ric)
  * standard: EZmock pk w/ systematics, w/ ric
  * standard_noric: EZmock pk w/ systematics, ric was subtracted using: standard_noric = standard - (ric - noric)
  * DR16: eBOSS DR16 data
  * DR16_noric: eBOSS DR16 data, ric was subtracted using: DR16_noric = DR16 - (ric - noric)
  * noric_cutsky: EZmock pk w/o systematics, footprint and redshift range have been cutted
    * effective redshifts:
      * LRG:
        * NGC: 0.692
        * SGC: 0.716
      * ELG:
        * NGC: 0.831
        * SGC: 0.823
      * X:
        * NGC: 0.761
        * SGC: 0.775
* window functions
  * measured using DR16 random catalogue and noric random catalogue
  * curves below 20 Mpc/h are smoothed by Savitzky–Golay filter
  * originally measured in 100 s bins from 1 to 3500 Mpc/h, and then cubic interpolated in logs space
  * normalized to match power spectrum in each folder
* naming convention:
  * P: power spectrum
  * Q: chained power spectrum
  * E: ELG
  * L: LRG
  * X: LRG ELG cross correlation
  * cov_: covariance matrix
  * win_: window function