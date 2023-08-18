# Distributional RL for Navigation
# Install OMPL
Link: https://ompl.kavrakilab.org/installation.html
change ompl-1.6.0/py-bindings/generate_bindings.py line 194 to:
try:
    self.ompl_ns.class_(f'SpecificParam< {self.string_decl} >').rename('SpecificParamString')
except:
    self.ompl_ns.class_(f'SpecificParam< std::basic_string< char > >').rename('SpecificParamString')

change install-ompl-ubuntu.sh line line 75 & 78 to : 
    # wget -O - https://github.com/ompl/${OMPL}/archive/1.6.0.tar.gz | tar zxf -