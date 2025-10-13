## Discovery Utilities

The `hspipy.utils.discovery` module provides utilities to automatically discover estimators, displays, and functions in the `hspipy` package.

### List all estimators

```python
from hspipy.utils.discovery import all_estimators

estimators = all_estimators()
print(estimators)
# Output: [('HSP', <class 'hspipy.hsp.HSP'>), ('HSPEstimator', <class 'hspipy.core.hsp_estimator.HSPEstimator'>)]
```

### List all displays

A *display* is a class intended for visualization or reporting of results (for example, plotting metrics or showing summaries).  
Currently, `hspipy` does not provide any display classes. Plotting is handled directly by methods on the estimator (e.g., `HSP.plot_2d()` and `HSP.plot_3d()`).  
Display classes may be added in the future to provide a more modular and extensible plotting interface.

```python
from hspipy.utils.discovery import all_displays

displays = all_displays()
print(displays)
# Output: []
```

### List all functions

```python
from hspipy.utils.discovery import all_functions

functions = all_functions()
print(functions)
# Output: [('WireframeSphere', <function WireframeSphere at 0x00000265F870E020>), ('compute_datafit', <function compute_datafit at 0x00000265ADEC3CE0>), ...]
```