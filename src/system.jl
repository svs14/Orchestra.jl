# System module.
module System

import PyCall: pyimport, pycall

export HAS_SKL,
       HAS_CRT

function check_py_dep(package::String)
  is_available = true
  try
    pyimport(package)
  catch
    is_available = false
  end
  return is_available
end

function check_r_dep(package::String)
  is_available = true
  try
    rpy2_packages = pyimport("rpy2.robjects.packages")
    pycall(rpy2_packages["importr"], Any, package)
  catch
    is_available = false
  end
  return is_available
end

# Check system for python dependencies.
HAS_SKL = check_py_dep("sklearn")
HAS_CRT = check_r_dep("caret")

end # module
