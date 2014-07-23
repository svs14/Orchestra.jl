# System module.
module System

import PyCall: pyimport, pycall

export LIB_SKL_AVAILABLE,
       LIB_CRT_AVAILABLE

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
    pycall(pyimport("rpy2.robjects.packages")["importr"], Any, package)
  catch
    is_available = false
  end
  return is_available
end

# Check system for python dependencies.
LIB_SKL_AVAILABLE = check_py_dep("sklearn")
LIB_CRT_AVAILABLE = check_r_dep("caret")

end # module
