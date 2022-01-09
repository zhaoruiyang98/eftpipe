<TeXmacs|2.1>

<style|<tuple|article|british>>

<\body>
  <doc-data|<doc-title|Notes of EFT LSS Multi-tracer>|<doc-author|<author-data|<author-name|Xiaoyong
  Mu>|<\author-affiliation>
    National Astronamical Obeservatories, Chinese Academy of Sciences

    \;

    School of Astronomy and Space Science, University of Chinese Academy of
    Sciences
  </author-affiliation>>>|<doc-date|<date>>>

  <abstract-data|<abstract|Notes of EFT LSS Multi-tracer.>>

  <section|Biased single-tracer in redshift space>

  Expand SPT kernals up to 3rd-order with counter-terms and stochastic terms:

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<delta\><rsub|A,r>>|<cell|=>|<cell|\<delta\><rsup|<around*|(|1|)>><rsub|<rsup|>A,r>+\<delta\><rsup|<around*|(|2|)>><rsub|<rsup|>A,r>+\<delta\><rsup|<around*|(|3|)>><rsub|<rsup|>A,r>+\<delta\><rsup|<around*|(|3,ct|)>><rsub|<rsup|>A,r>+\<delta\><rsup|<around*|(|\<varepsilon\>|)>><rsub|<rsup|>A,r><eq-number>>>>>
  </eqnarray*>

  where\ 

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<delta\><rsub|A,r><rsup|<around*|(|n|)>><around*|(|<wide|k|\<vect\>>|)>>|<cell|=>|<cell|<big|int>d<rsup|3>q<rsub|1>\<ldots\>d<rsup|3>q<rsub|n>K<rsub|A,r><rsup|<around*|(|n|)>><around*|(|<wide|q<rsub|1><rsub|>|\<vect\>>,\<ldots\>,<wide|q|\<vect\>><rsub|n>|)><rsub|sym>\<delta\><rsup|3><rsub|D><around*|(|<wide|k|\<vect\>>-<wide|q|\<vect\>><rsub|1>\<ldots\>-<wide|q|\<vect\>><rsub|n>|)>\<delta\><rsup|<around*|(|1|)>><around*|(|<wide|q|\<vect\>><rsub|1>|)>\<ldots\>\<delta\><rsup|<around*|(|1|)>><around*|(|<wide|q|\<vect\>><rsub|n>|)><eq-number>>>>>
  </eqnarray*>

  The kernal <math|K<rsub|A,r><rsup|<around*|(|1|)>><around*|(|<wide|q|\<vect\>><rsub|1>|)>,K<rsub|A,r><rsup|<around*|(|2|)>><around*|(|<wide|q|\<vect\>><rsub|1>,<wide|q|\<vect\>><rsub|2>|)>,K<rsub|A,r><rsup|<around*|(|3|)>><around*|(|<wide|q|\<vect\>><rsub|1>,<wide|q|\<vect\>><rsub|2>,<wide|q|\<vect\>><rsub|3>|)>
  ><math|>have been expressed with real space kernals.

  The counter-terms are as follows:

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<delta\><rsup|<around*|(|3,ct|)>><rsub|<rsup|>A,r>>|<cell|=>|<cell|<around*|(|c<rsup|A
    ><rsub|ct>+c<rsup|A ><rsub|r,1>\<mu\><rsup|2>+c<rsup|
    ><rsub|r,2>\<mu\><rsup|4>|)>k<rsup|2>
    \<delta\><rsup|<around*|(|1|)>><eq-number>>>>>
  </eqnarray*>

  where the RSD counter-term <math|c<rsub|r,2>> is independent of bia of the
  tracer A.

  The stochasitic terms are as follows:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<around*|\<langle\>|\<delta\><rsup|><rsub|A,r>\<delta\><rsup|><rsub|A,r>|\<rangle\>>>|<cell|=>|<cell|<frac|1|<wide|n|\<bar\>><rsub|A>><around*|(|c<rsup|A><rsub|\<varepsilon\>,1>+c<rsup|A><rsub|\<varepsilon\>,2>k<rsup|2>+c<rsup|A><rsub|\<varepsilon\>,3>f\<mu\><rsup|2>k<rsup|2>|)><eq-number>>>>>
  </eqnarray*>

  \;

  Finally we get the single-tracer power spectrum <math|P<rsub|A A>>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<around*|\<langle\>|\<delta\><rsub|A,r><rsup|><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsub|A,r><rsup|><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>>>|<cell|=>|<cell|<around*|\<langle\>|\<delta\><rsub|A,r><rsup|<around*|(|1|)>><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsub|A,r><rsup|<around*|(|1|)>><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>>+<around*|\<langle\>|\<delta\><rsub|A,r><rsup|<around*|(|2|)>><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsub|A,r><rsup|<around*|(|2|)>><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>>+2<around*|\<langle\>|\<delta\><rsub|A,r><rsup|<around*|(|1|)>><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsub|A,r><rsup|<around*|(|3|)>><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>><eq-number>>>|<row|<cell|>|<cell|>|<cell|+2<around*|\<langle\>|\<delta\><rsup|<around*|(|1|)>><rsub|A,r><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsup|<around*|(|3,ct|)>><rsub|A,r><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>>+<around*|\<langle\>|\<delta\><rsup|<around*|(|\<varepsilon\>|)>><rsub|A,r><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsup|<around*|(|\<varepsilon\>|)>><rsub|A,r><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>>>>|<row|<cell|>|<cell|=>|<cell|<around*|(|K<rsup|<around*|(|1|)>><rsub|A,r>|)><rsup|2>P<rsub|11><around*|(|k|)>+2<big|int>d<rsup|3>q<around*|(|K<rsub|A,r><rsup|<around*|(|2|)>><around*|(|<wide|q|\<vect\>><rsub|1>,<wide|k|\<vect\>>-<wide|q|\<vect\>><rsub|>|)><rsub|sym>|)><rsup|2>P<rsub|11><around*|(|<around*|\||<wide|k|\<vect\>>-<wide|q|\<vect\>>|\|>|)>P<rsub|11><around*|(|q|)>>>|<row|<cell|>|<cell|>|<cell|+6<big|int>d<rsup|3><wide|q|\<vect\>>K<rsup|<around*|(|3|)>><rsub|A,r><around*|(|<wide|q|\<vect\>>,-<wide|q|\<vect\>>,<wide|k|\<vect\>><rsub|>|)><rsub|sym>K<rsub|A,r><rsup|<around*|(|1|)>>P<rsub|11><around*|(|q|)>P<rsub|11><around*|(|k|)>>>|<row|<cell|>|<cell|>|<cell|+2K<rsup|<around*|(|1|)>><rsub|A,r>P<rsub|11><around*|(|k|)><rsup|><around*|(|c<rsup|A
    ><rsub|ct>+c<rsup|A ><rsub|r,1>\<mu\><rsup|2>+c<rsup|
    ><rsub|r,2>\<mu\><rsup|4>|)>k<rsup|2>>>|<row|<cell|>|<cell|>|<cell|+<frac|1|<wide|n|\<bar\>><rsub|A>><around*|(|c<rsup|A><rsub|\<varepsilon\>,1>+c<rsup|A><rsub|\<varepsilon\>,2>k<rsup|2>+c<rsup|A><rsub|\<varepsilon\>,3>f\<mu\><rsup|2>k<rsup|2>|)>>>>>
  </eqnarray*>

  <section|Biased multi-tracer in redshift space>

  Assuming the stochasitic terms of the two tracers are uncorrelated with
  Gaussian initial conditions, we have:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<around*|\<langle\>|\<delta\><rsup|><rsub|A,r>\<delta\><rsup|><rsub|B,r>|\<rangle\>>>|<cell|=>|<cell|<frac|1|2><around*|(|<frac|1|<wide|n|\<bar\>><rsub|A>>+<frac|1|<wide|n|\<bar\>><rsub|B>>|)><around*|(|c<rsup|A
    B><rsub|\<varepsilon\>,1>+c<rsup|A B><rsub|\<varepsilon\>,2><around*|(|<frac|k|k<rsub|M>>|)><rsup|2>+c<rsup|A
    B><rsub|\<varepsilon\>,3>f\<mu\><rsup|2><around*|(|<frac|k|k<rsub|M>>|)><rsup|2>|)><eq-number>>>>>
  </eqnarray*>

  Therefore, we can write the cross power spectrum <math|P<rsub|A B>> in this
  way:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<around*|\<langle\>|\<delta\><rsub|A,r><rsup|><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsub|B,r><rsup|><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>>>|<cell|=>|<cell|<around*|\<langle\>|\<delta\><rsub|A,r><rsup|<around*|(|1|)>><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsub|A,r><rsup|<around*|(|1|)>><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>>+<around*|\<langle\>|\<delta\><rsub|A,r><rsup|<around*|(|2|)>><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsub|A,r><rsup|<around*|(|2|)>><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>>+<around*|\<langle\>|\<delta\><rsub|A,r><rsup|<around*|(|1|)>><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsub|B,r><rsup|<around*|(|3|)>><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>>+<around*|\<langle\>|\<delta\><rsub|A,r><rsup|<around*|(|3|)>><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsub|B,r><rsup|<around*|(|1|)>><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>>>>|<row|<cell|>|<cell|>|<cell|+<around*|\<langle\>|\<delta\><rsup|<around*|(|1|)>><rsub|B,r><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsup|<around*|(|3,ct|)>><rsub|A,r><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>>+<around*|\<langle\>|\<delta\><rsup|<around*|(|1|)>><rsub|A,r><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsup|<around*|(|3,ct|)>><rsub|B,r><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>>+<around*|\<langle\>|\<delta\><rsup|<around*|(|\<varepsilon\>|)>><rsub|A,r><around*|(|<wide|k|\<vect\>>|)>\<delta\><rsup|<around*|(|\<varepsilon\>|)>><rsub|B,r><around*|(|<wide|k|\<vect\>>|)>|\<rangle\>><eq-number>>>|<row|<cell|>|<cell|=>|<cell|K<rsup|<around*|(|1|)>><rsub|A,r>K<rsup|<around*|(|1|)>><rsub|B,r>P<rsub|11><around*|(|k|)>+2<big|int>d<rsup|3><wide|q|\<vect\>>K<rsup|<around*|(|2|)>><rsub|A,r><around*|(|<wide|q|\<vect\>>,<wide|k|\<vect\>>-<wide|q|\<vect\>><rsub|>|)><rsub|sym>K<rsup|<around*|(|2|)>><rsub|B,r><around*|(|<wide|q|\<vect\>>,<wide|k|\<vect\>>-<wide|q|\<vect\>><rsub|>|)><rsub|sym>P<rsub|11><around*|(|<around*|\||<wide|k|\<vect\>>-<wide|q|\<vect\>>|\|>|)>P<rsub|11><around*|(|q|)>>>|<row|<cell|>|<cell|>|<cell|+3<big|int>d<rsup|3><wide|q|\<vect\>>K<rsup|<around*|(|3|)>><rsub|B,r><around*|(|<wide|q|\<vect\>>,-<wide|q|\<vect\>>,<wide|k|\<vect\>><rsub|>|)><rsub|sym><rsub|><rsub|>K<rsup|<around*|(|1|)>><rsub|A,r>P<rsub|11><around*|(|q|)>P<rsub|11><around*|(|k|)>>>|<row|<cell|>|<cell|>|<cell|+3<big|int>d<rsup|3><wide|q|\<vect\>>K<rsup|<around*|(|3|)>><rsub|A,r><around*|(|<wide|q|\<vect\>>,-<wide|q|\<vect\>>,<wide|k|\<vect\>><rsub|>|)><rsub|sym><rsub|>K<rsup|<around*|(|1|)>><rsub|B,r>P<rsub|11><around*|(|q|)>P<rsub|11><around*|(|k|)>>>|<row|<cell|>|<cell|>|<cell|+K<rsup|<around*|(|1|)>><rsub|B,r>P<rsub|11><around*|(|k|)><rsup|><around*|(|c<rsup|A
    ><rsub|ct>+c<rsup|A ><rsub|r,1>\<mu\><rsup|2>+c<rsup|
    ><rsub|r,2>\<mu\><rsup|4>|)>k<rsup|2>>>|<row|<cell|>|<cell|>|<cell|+K<rsup|<around*|(|1|)>><rsub|A,r>P<rsub|11><around*|(|k|)><rsup|><around*|(|c<rsup|
    B><rsub|ct>+c<rsup|B><rsub|r,1>\<mu\><rsup|2>+c<rsup|><rsub|r,2>\<mu\><rsup|4>|)>k<rsup|2>>>|<row|<cell|>|<cell|>|<cell|+<frac|1|2><around*|(|<frac|1|<wide|n|\<bar\>><rsub|A>>+<frac|1|<wide|n|\<bar\>><rsub|B>>|)><around*|(|c<rsup|A
    B><rsub|\<varepsilon\>,1>+c<rsup|A B><rsub|\<varepsilon\>,2><around*|(|<frac|k|k<rsub|M>>|)><rsup|2>+c<rsup|A
    B><rsub|\<varepsilon\>,3>f\<mu\><rsup|2><around*|(|<frac|k|k<rsub|M>>|)><rsup|2>|)>>>>>
  </eqnarray*>

  For more clear expression, we organize all of the parameters needed:

  <\eqnarray*>
    <tformat|<table|<row|<cell|P<rsub|A A>>|<cell|:>|<cell|<around*|{|b<rsup|A><rsub|1>,b<rsup|A><rsub|2>,b<rsup|A><rsub|3>,b<rsup|A><rsub|4>,c<rsup|A><rsub|ct>,c<rsup|A><rsub|r,1>,c<rsup|><rsub|r,2>,c<rsup|A><rsub|\<varepsilon\>,1>,c<rsup|A><rsub|\<varepsilon\>,2>,c<rsup|A><rsub|\<varepsilon\>,3>|}><eq-number>>>|<row|<cell|P<rsub|B
    B>>|<cell|:>|<cell|<around*|{|b<rsup|B><rsub|1>,b<rsup|B><rsub|2>,b<rsup|B><rsub|3>,b<rsup|B><rsub|4>,c<rsup|B><rsub|ct>,c<rsup|B><rsub|r,1>,c<rsup|><rsub|r,2>,c<rsup|B><rsub|\<varepsilon\>,1>,c<rsup|B><rsub|\<varepsilon\>,2>,c<rsup|B><rsub|\<varepsilon\>,3>|}>>>|<row|<cell|P<rsub|AB>>|<cell|:>|<cell|<around*|{|b<rsup|A><rsub|1>,b<rsup|A><rsub|2>,b<rsup|A><rsub|3>,b<rsup|A><rsub|4>,b<rsup|B><rsub|1>,b<rsup|B><rsub|2>,b<rsup|B><rsub|3>,b<rsup|B><rsub|4>,c<rsup|A><rsub|ct>,c<rsup|A><rsub|r,1>,c<rsup|B><rsub|ct>,c<rsup|B><rsub|r,1>,c<rsup|><rsub|r,2>,c<rsup|A
    B><rsub|\<varepsilon\>,1>,c<rsup|A B><rsub|\<varepsilon\>,2>,c<rsup|A
    B><rsub|\<varepsilon\>,3>|}>>>>>
  </eqnarray*>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1|..\\..\\..\\AppData\\Roaming\\TeXmacs\\texts\\scratch\\no_name_3.tm>>
    <associate|auto-2|<tuple|2|1|..\\..\\..\\AppData\\Roaming\\TeXmacs\\texts\\scratch\\no_name_3.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Biased
      single-tracer in redshift space> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Biased
      multi-tracer in redshift space> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>