# Compressed LLC Reader

## Usage Examples

### Pleaides

```python

from llcreader import open_llc_dataset
from llcreader.stores import PleiadesFilesytem

ds = open_llc_dataset(model='llc_4320', store=PleiadesFilesytem(),
                      variables=['Salt', 'Theta'])
```

### NAS ECCO HTTP Portal

```python

from llcreader import open_llc_dataset
from llcreader.stores import NasEccoHttpPortal

ds = open_llc_dataset(model='llc_4320', store=NasEccoHttpPortal(),
                      variables=['Salt', 'Theta'])
```

### Pangeo Google Cloud Storage

```python

from llcreader import open_llc_dataset
from llcreader.stores import PangeoZarrGCS

ds = open_llc_dataset(model='llc_4320', store=PangeoZarrGCS(),
                      variables=['Salt', 'Theta'])
```
