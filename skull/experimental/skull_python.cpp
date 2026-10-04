// ============================================================
//  SKULL PYTHON BINDINGS v1.0.0
//  Python C API bindings for Skull
// ============================================================

#include <Python.h>
#include "tensor.h"
#include "model.h"
#include "optimizer.h"
#include "layers.h"
#include "tokenizer.h"
#include "config.h"

// ============================================================
//  PYTHON MODULE DEFINITION
// ============================================================

// Forward declarations
static PyObject* SkullModule_init(PyObject* self, PyObject* args);
static PyObject* SkullTensor_new(PyObject* self, PyObject* args);
static PyObject* SkullTensor_data(PyObject* self, PyObject* args);
static PyObject* SkullTensor_shape(PyObject* self, PyObject* args);
static PyObject* SkullTensor_print(PyObject* self, PyObject* args);

static PyObject* SkullModel_new(PyObject* self, PyObject* args);
static PyObject* SkullModel_train(PyObject* self, PyObject* args);
static PyObject* SkullModel_generate(PyObject* self, PyObject* args);
static PyObject* SkullModel_save(PyObject* self, PyObject* args);
static PyObject* SkullModel_load(PyObject* self, PyObject* args);

static PyObject* SkullTokenizer_new(PyObject* self, PyObject* args);
static PyObject* SkullTokenizer_encode(PyObject* self, PyObject* args);
static PyObject* SkullTokenizer_decode(PyObject* self, PyObject* args);

static PyObject* skull_train(PyObject* self, PyObject* args);
static PyObject* skull_generate(PyObject* self, PyObject* args);
static PyObject* skull_info(PyObject* self, PyObject* args);

// ============================================================
//  PYTHON TENSOR OBJECT
// ============================================================

typedef struct {
    PyObject_HEAD
    TensorPtr tensor;
} SkullTensorObject;

static PyObject* SkullTensor_new(PyObject* self, PyObject* args) {
    int rows, cols;
    float init = 0.0f;
    
    if (!PyArg_ParseTuple(args, "ii|f", &rows, &cols, &init)) {
        return NULL;
    }
    
    SkullTensorObject* obj = (SkullTensorObject*)PyObject_New(PyObject, &PyTypeObject);
    if (!obj) return NULL;
    
    obj->tensor = std::make_shared<Tensor>(rows, cols, init);
    
    return (PyObject*)obj;
}

static PyObject* SkullTensor_data(PyObject* self, PyObject* args) {
    SkullTensorObject* obj = (SkullTensorObject*)self;
    
    PyObject* list = PyList_New(obj->tensor->data.size());
    for (size_t i = 0; i < obj->tensor->data.size(); ++i) {
        PyList_SetItem(list, i, PyFloat_FromDouble(obj->tensor->data[i]));
    }
    
    return list;
}

static PyObject* SkullTensor_shape(PyObject* self, PyObject* args) {
    SkullTensorObject* obj = (SkullTensorObject*)self;
    
    PyObject* tuple = PyTuple_New(2);
    PyTuple_SetItem(tuple, 0, PyLong_FromLong(obj->tensor->rows));
    PyTuple_SetItem(tuple, 1, PyLong_FromLong(obj->tensor->cols));
    
    return tuple;
}

static PyObject* SkullTensor_print(PyObject* self, PyObject* args) {
    SkullTensorObject* obj = (SkullTensorObject*)self;
    obj->tensor->print();
    Py_RETURN_NONE;
}

static void SkullTensor_dealloc(PyObject* self) {
    SkullTensorObject* obj = (SkullTensorObject*)self;
    obj->tensor.reset();
    PyObject_Del(self);
}

static PyMethodDef SkullTensor_methods[] = {
    {"data", SkullTensor_data, METH_NOARGS, "Get tensor data as list"},
    {"shape", SkullTensor_shape, METH_NOARGS, "Get tensor shape"},
    {"print", SkullTensor_print, METH_NOARGS, "Print tensor"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject SkullTensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "skull.Tensor",
    sizeof(SkullTensorObject),
    0,
    SkullTensor_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    SkullTensor_methods,
    0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Skull Tensor",
    0, 0, 0, 0, 0, 0, 0,
    SkullTensor_new,
};

// ============================================================
//  PYTHON MODEL OBJECT
// ============================================================

typedef struct {
    PyObject_HEAD
    ModelPtr model;
} SkullModelObject;

static PyObject* SkullModel_new(PyObject* self, PyObject* args) {
    const char* name = "MyModel";
    const char* architecture = "transformer";
    int vocab_size = 256;
    int dim = 512;
    int num_layers = 6;
    int num_heads = 8;
    
    if (!PyArg_ParseTuple(args, "|ssiiii", &name, &architecture, &vocab_size, &dim, &num_layers, &num_heads)) {
        return NULL;
    }
    
    SkullModelObject* obj = (SkullModelObject*)PyObject_New(PyObject, &PyTypeObject);
    if (!obj) return NULL;
    
    try {
        ModelArchitecture arch;
        if (strcmp(architecture, "feedforward") == 0 || strcmp(architecture, "FF") == 0) {
            arch = ModelArchitecture::FEEDFORWARD;
        } else if (strcmp(architecture, "transformer") == 0) {
            arch = ModelArchitecture::TRANSFORMER;
        } else if (strcmp(architecture, "lstm") == 0) {
            arch = ModelArchitecture::LSTM;
        } else {
            arch = ModelArchitecture::FEEDFORWARD;
        }
        
        obj->model = ModelFactory::create(arch, vocab_size, dim, name, num_layers, num_heads);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    
    return (PyObject*)obj;
}

static PyObject* SkullModel_train(PyObject* self, PyObject* args) {
    SkullModelObject* obj = (SkullModelObject*)self;
    
    const char* data_path;
    int epochs = 10;
    float lr = 0.001f;
    
    if (!PyArg_ParseTuple(args, "s|if", &data_path, &epochs, &lr)) {
        return NULL;
    }
    
    try {
        // Set learning rate
        obj->model->optimizer->set_learning_rate(lr);
        
        // Train (simplified - actual training would need proper data loading)
        std::vector<std::string> train_files = {data_path};
        train_model(obj->model, train_files, epochs);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject* SkullModel_generate(PyObject* self, PyObject* args) {
    SkullModelObject* obj = (SkullModelObject*)self;
    
    const char* prompt = "";
    int max_tokens = 100;
    float temperature = 0.8f;
    
    if (!PyArg_ParseTuple(args, "|sif", &prompt, &max_tokens, &temperature)) {
        return NULL;
    }
    
    try {
        std::string text = obj->model->generate_text(prompt, max_tokens, temperature);
        return PyUnicode_FromString(text.c_str());
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject* SkullModel_save(PyObject* self, PyObject* args) {
    SkullModelObject* obj = (SkullModelObject*)self;
    
    const char* filepath;
    if (!PyArg_ParseTuple(args, "s", &filepath)) {
        return NULL;
    }
    
    try {
        obj->model->save(filepath);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject* SkullModel_load(PyObject* self, PyObject* args) {
    SkullModelObject* obj = (SkullModelObject*)self;
    
    const char* filepath;
    if (!PyArg_ParseTuple(args, "s", &filepath)) {
        return NULL;
    }
    
    try {
        obj->model->load(filepath);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject* SkullModel_info(PyObject* self, PyObject* args) {
    SkullModelObject* obj = (SkullModelObject*)self;
    
    try {
        std::string info = obj->model->info();
        return PyUnicode_FromString(info.c_str());
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static void SkullModel_dealloc(PyObject* self) {
    SkullModelObject* obj = (SkullModelObject*)self;
    obj->model.reset();
    PyObject_Del(self);
}

static PyMethodDef SkullModel_methods[] = {
    {"train", SkullModel_train, METH_VARARGS, "Train the model"},
    {"generate", SkullModel_generate, METH_VARARGS, "Generate text"},
    {"save", SkullModel_save, METH_VARARGS, "Save model"},
    {"load", SkullModel_load, METH_VARARGS, "Load model"},
    {"info", SkullModel_info, METH_NOARGS, "Get model info"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject SkullModelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "skull.Model",
    sizeof(SkullModelObject),
    0,
    SkullModel_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    SkullModel_methods,
    0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Skull Model",
    0, 0, 0, 0, 0, 0, 0,
    SkullModel_new,
};

// ============================================================
//  PYTHON TOKENIZER OBJECT
// ============================================================

typedef struct {
    PyObject_HEAD
    TokenizerPtr tokenizer;
} SkullTokenizerObject;

static PyObject* SkullTokenizer_new(PyObject* self, PyObject* args) {
    int vocab_size = 256;
    const char* bpe_path = NULL;
    
    if (!PyArg_ParseTuple(args, "|is", &vocab_size, &bpe_path)) {
        return NULL;
    }
    
    SkullTokenizerObject* obj = (SkullTokenizerObject*)PyObject_New(PyObject, &PyTypeObject);
    if (!obj) return NULL;
    
    try {
        if (bpe_path) {
            obj->tokenizer = std::make_shared<BPETokenizer>(vocab_size, bpe_path);
        } else {
            obj->tokenizer = std::make_shared<BPETokenizer>(vocab_size);
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    
    return (PyObject*)obj;
}

static PyObject* SkullTokenizer_encode(PyObject* self, PyObject* args) {
    SkullTokenizerObject* obj = (SkullTokenizerObject*)self;
    
    const char* text;
    if (!PyArg_ParseTuple(args, "s", &text)) {
        return NULL;
    }
    
    try {
        auto tokens = obj->tokenizer->encode(text);
        PyObject* list = PyList_New(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            PyList_SetItem(list, i, PyLong_FromLong(tokens[i]));
        }
        return list;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject* SkullTokenizer_decode(PyObject* self, PyObject* args) {
    SkullTokenizerObject* obj = (SkullTokenizerObject*)self;
    
    PyObject* tokens_list;
    if (!PyArg_ParseTuple(args, "O", &tokens_list)) {
        return NULL;
    }
    
    try {
        std::vector<int> tokens;
        Py_ssize_t size = PyList_Size(tokens_list);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* item = PyList_GetItem(tokens_list, i);
            tokens.push_back(PyLong_AsLong(item));
        }
        
        std::string text = obj->tokenizer->decode(tokens);
        return PyUnicode_FromString(text.c_str());
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static void SkullTokenizer_dealloc(PyObject* self) {
    SkullTokenizerObject* obj = (SkullTokenizerObject*)self;
    obj->tokenizer.reset();
    PyObject_Del(self);
}

static PyMethodDef SkullTokenizer_methods[] = {
    {"encode", SkullTokenizer_encode, METH_VARARGS, "Encode text to tokens"},
    {"decode", SkullTokenizer_decode, METH_VARARGS, "Decode tokens to text"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject SkullTokenizerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "skull.Tokenizer",
    sizeof(SkullTokenizerObject),
    0,
    SkullTokenizer_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    SkullTokenizer_methods,
    0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Skull Tokenizer",
    0, 0, 0, 0, 0, 0, 0,
    SkullTokenizer_new,
};

// ============================================================
//  MODULE FUNCTIONS
// ============================================================

static PyObject* skull_init(PyObject* self, PyObject* args) {
    skull_init_config();
    Py_RETURN_NONE;
}

static PyObject* skull_train(PyObject* self, PyObject* args) {
    const char* config_file;
    
    if (!PyArg_ParseTuple(args, "s", &config_file)) {
        return NULL;
    }
    
    try {
        // Load config and train (simplified)
        std::cout << "Training with config: " << config_file << std::endl;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject* skull_generate(PyObject* self, PyObject* args) {
    const char* model_path;
    const char* prompt = "";
    int max_tokens = 100;
    float temperature = 0.8f;
    
    if (!PyArg_ParseTuple(args, "s|sif", &model_path, &prompt, &max_tokens, &temperature)) {
        return NULL;
    }
    
    try {
        // Load model and generate (simplified)
        std::cout << "Generating text with model: " << model_path << std::endl;
        std::cout << "Prompt: " << prompt << std::endl;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    
    return PyUnicode_FromString("");
}

static PyObject* skull_info(PyObject* self, PyObject* args) {
    std::string info = skull_get_build_info();
    return PyUnicode_FromString(info.c_str());
}

static PyObject* skull_version(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(skull_get_version());
}

// ============================================================
//  MODULE DEFINITION
// ============================================================

static PyMethodDef SkullMethods[] = {
    {"init", skull_init, METH_NOARGS, "Initialize Skull"},
    {"train", skull_train, METH_VARARGS, "Train a model"},
    {"generate", skull_generate, METH_VARARGS, "Generate text"},
    {"info", skull_info, METH_NOARGS, "Get Skull info"},
    {"version", skull_version, METH_NOARGS, "Get Skull version"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef SkullModule = {
    PyModuleDef_HEAD_INIT,
    "skull",
    "Skull - A programming language for training language models",
    -1,
    SkullMethods,
};

// ============================================================
//  MODULE INITIALIZATION
// ============================================================

PyMODINIT_FUNC PyInit_skull(void) {
    PyObject* m;
    
    // Initialize types
    if (PyType_Ready(&SkullTensorType) < 0) return NULL;
    if (PyType_Ready(&SkullModelType) < 0) return NULL;
    if (PyType_Ready(&SkullTokenizerType) < 0) return NULL;
    
    // Create module
    m = PyModule_Create(&SkullModule);
    if (!m) return NULL;
    
    // Add types to module
    Py_INCREF(&SkullTensorType);
    PyModule_AddObject(m, "Tensor", (PyObject*)&SkullTensorType);
    
    Py_INCREF(&SkullModelType);
    PyModule_AddObject(m, "Model", (PyObject*)&SkullModelType);
    
    Py_INCREF(&SkullTokenizerType);
    PyModule_AddObject(m, "Tokenizer", (PyObject*)&SkullTokenizerType);
    
    // Add constants
    PyModule_AddIntConstant(m, "VERSION_MAJOR", SKULL_VERSION_MAJOR);
    PyModule_AddIntConstant(m, "VERSION_MINOR", SKULL_VERSION_MINOR);
    PyModule_AddIntConstant(m, "VERSION_PATCH", SKULL_VERSION_PATCH);
    
    // Initialize Skull
    skull_init_config();
    
    return m;
}
