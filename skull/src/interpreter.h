#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include "ast.h"
#include "tensor.h"
#include "trainer.h"
#include "generator.h"

// ============================================================
//  SKULL INTERPRETER  v0.7.0
//  Neu: gpu_info(), BPE-Tokenisierung, Multi-Format Training
// ============================================================

struct SkullValue {
    enum class Kind { NUMBER, STRING, BOOL, TENSOR, NOTHING } kind;
    double number=0.0; std::string text; bool flag=false; TensorPtr tensor;
    SkullValue():kind(Kind::NOTHING){}
    SkullValue(double v):kind(Kind::NUMBER),number(v){}
    SkullValue(const std::string& s):kind(Kind::STRING),text(s){}
    SkullValue(bool b):kind(Kind::BOOL),flag(b){}
    SkullValue(TensorPtr t):kind(Kind::TENSOR),tensor(t){}
    bool is_truthy() const {
        if(kind==Kind::NUMBER)return number!=0.0;
        if(kind==Kind::BOOL)return flag;
        if(kind==Kind::STRING)return !text.empty();
        if(kind==Kind::TENSOR)return tensor!=nullptr;
        return false;
    }
    double as_number(int line=0) const {
        if(kind==Kind::NUMBER)return number;
        if(kind==Kind::BOOL)return flag?1.0:0.0;
        if(kind==Kind::TENSOR&&tensor&&tensor->rows==1&&tensor->cols==1)return tensor->data[0];
        throw std::runtime_error("Zeile "+std::to_string(line)+": Zahl erwartet");
    }
    TensorPtr as_tensor(int line=0) const {
        if(kind==Kind::TENSOR)return tensor;
        if(kind==Kind::NUMBER){auto t=std::make_shared<Tensor>(1,1);t->data[0]=number;return t;}
        throw std::runtime_error("Zeile "+std::to_string(line)+": Tensor erwartet");
    }
    std::string to_string() const {
        if(kind==Kind::NUMBER){
            if(number==std::floor(number)&&std::abs(number)<1e15)return std::to_string((long long)number);
            std::string s=std::to_string(number);s.erase(s.find_last_not_of('0')+1);
            if(s.back()=='.')s+="0";return s;
        }
        if(kind==Kind::STRING)return text;
        if(kind==Kind::BOOL)return flag?"true":"false";
        if(kind==Kind::TENSOR&&tensor)return tensor->to_string();
        return "";
    }
    void print() const {
        if(kind==Kind::TENSOR&&tensor)tensor->print();
        else std::cout<<to_string();
    }
};

struct ReturnSignal{SkullValue value;};

struct Environment {
    std::map<std::string,SkullValue> vars;
    std::shared_ptr<Environment> parent;
    Environment()=default;
    explicit Environment(std::shared_ptr<Environment> p):parent(p){}
    void define(const std::string& n,SkullValue v){vars[n]=std::move(v);}
    void assign(const std::string& n,SkullValue v){
        if(vars.count(n)){vars[n]=std::move(v);return;}
        if(parent&&parent->has(n)){parent->assign(n,std::move(v));return;}
        vars[n]=std::move(v);
    }
    SkullValue get(const std::string& n,int line=0)const{
        auto it=vars.find(n);if(it!=vars.end())return it->second;
        if(parent)return parent->get(n,line);
        throw std::runtime_error("Zeile "+std::to_string(line)+": Unbekannte Variable '"+n+"'");
    }
    bool has(const std::string& n)const{
        if(vars.count(n))return true;if(parent)return parent->has(n);return false;
    }
};

struct SkullFunction {
    std::string name;
    std::vector<std::string> params;
    std::vector<std::unique_ptr<StmtNode>>* body;
    std::shared_ptr<Environment> closure;
};

class Interpreter {
private:
    std::shared_ptr<Environment> global_env;
    std::map<std::string,SkullFunction> functions;
    std::map<std::string,ModelStmt*> models;

    SkullValue eval_expr(const ExprNode* node,std::shared_ptr<Environment> env){
        if(!node)throw std::runtime_error("Leerer Ausdruck");
        if(auto*n=dynamic_cast<const NumberExpr*>(node))return SkullValue(n->value);
        if(auto*n=dynamic_cast<const StringExpr*>(node))return SkullValue(n->value);
        if(auto*n=dynamic_cast<const BoolExpr*>(node))return SkullValue(n->value);
        if(auto*n=dynamic_cast<const IdentExpr*>(node))return env->get(n->name,n->line);
        if(auto*n=dynamic_cast<const BinaryExpr*>(node))return eval_binary(n,env);
        if(auto*n=dynamic_cast<const CallExpr*>(node))return eval_call(n,env);
        throw std::runtime_error("Unbekannter Ausdruck");
    }

    SkullValue eval_binary(const BinaryExpr*n,std::shared_ptr<Environment>env){
        SkullValue lv=eval_expr(n->left.get(),env),rv=eval_expr(n->right.get(),env);
        const std::string&op=n->op;
        bool lT=(lv.kind==SkullValue::Kind::TENSOR),rT=(rv.kind==SkullValue::Kind::TENSOR);
        if(op=="*"&&(lT||rT)){
            if(lv.kind==SkullValue::Kind::NUMBER)return SkullValue(tensor_scale(rv.as_tensor(n->line),lv.number));
            if(rv.kind==SkullValue::Kind::NUMBER)return SkullValue(tensor_scale(lv.as_tensor(n->line),rv.number));
            return SkullValue(tensor_matmul(lv.as_tensor(n->line),rv.as_tensor(n->line)));
        }
        if(op=="+"&&(lT||rT))return SkullValue(tensor_add(lv.as_tensor(n->line),rv.as_tensor(n->line)));
        if(op=="-"&&(lT||rT))return SkullValue(tensor_sub(lv.as_tensor(n->line),rv.as_tensor(n->line)));
        if(op=="+"&&(lv.kind==SkullValue::Kind::STRING||rv.kind==SkullValue::Kind::STRING))
            return SkullValue(lv.to_string()+rv.to_string());
        if(op=="==")return SkullValue(lv.to_string()==rv.to_string());
        if(op=="!=")return SkullValue(lv.to_string()!=rv.to_string());
        double l=lv.as_number(n->line),r=rv.as_number(n->line);
        if(op=="+")return SkullValue(l+r);if(op=="-")return SkullValue(l-r);
        if(op=="*")return SkullValue(l*r);
        if(op=="/"){if(r==0.0)throw std::runtime_error("Division durch 0");return SkullValue(l/r);}
        if(op=="<")return SkullValue(l<r);if(op==">")return SkullValue(l>r);
        if(op=="<=")return SkullValue(l<=r);if(op==">=")return SkullValue(l>=r);
        throw std::runtime_error("Unbekannter Operator '"+op+"'");
    }

    SkullValue eval_call(const CallExpr*n,std::shared_ptr<Environment>env){
        std::vector<SkullValue>args;
        for(const auto&arg:n->args)args.push_back(eval_expr(arg.get(),env));

        if(n->name=="print"){
            for(size_t i=0;i<args.size();++i){if(i>0)std::cout<<" ";args[i].print();}
            std::cout<<"\n";return SkullValue();
        }
        if(n->name=="skull_info"){print_skull_info();return SkullValue();}
        if(n->name=="gpu_info"){skull_print_gpu_status();return SkullValue();}

        if(n->name=="rand_tensor"){if(args.size()<2)throw std::runtime_error("rand_tensor() braucht 2 Argumente");return SkullValue(tensor_rand((size_t)args[0].as_number(n->line),(size_t)args[1].as_number(n->line)));}
        if(n->name=="zeros"){if(args.size()<2)throw std::runtime_error("zeros() braucht 2 Argumente");return SkullValue(tensor_zeros((size_t)args[0].as_number(n->line),(size_t)args[1].as_number(n->line)));}
        if(n->name=="ones"){if(args.size()<2)throw std::runtime_error("ones() braucht 2 Argumente");return SkullValue(tensor_ones((size_t)args[0].as_number(n->line),(size_t)args[1].as_number(n->line)));}
        if(n->name=="relu"){if(args.empty())throw std::runtime_error("relu() braucht 1 Argument");if(args[0].kind==SkullValue::Kind::TENSOR)return SkullValue(tensor_relu(args[0].tensor));double x=args[0].as_number(n->line);return SkullValue(x>0.0?x:0.0);}
        if(n->name=="sigmoid"){if(args.empty())throw std::runtime_error("sigmoid() braucht 1 Argument");if(args[0].kind==SkullValue::Kind::TENSOR)return SkullValue(tensor_sigmoid(args[0].tensor));double x=args[0].as_number(n->line);return SkullValue(1.0/(1.0+std::exp(-x)));}
        if(n->name=="tanh_act"){if(args.empty())throw std::runtime_error("tanh_act() braucht 1 Argument");if(args[0].kind==SkullValue::Kind::TENSOR)return SkullValue(tensor_tanh(args[0].tensor));return SkullValue(std::tanh(args[0].as_number(n->line)));}
        if(n->name=="mse_loss"){if(args.size()<2)throw std::runtime_error("mse_loss() braucht 2 Argumente");return SkullValue(tensor_mse_loss(args[0].as_tensor(n->line),args[1].as_tensor(n->line)));}
        if(n->name=="backward"){if(args.empty())throw std::runtime_error("backward() braucht 1 Argument");args[0].as_tensor(n->line)->backward();return SkullValue();}
        if(n->name=="update"){if(args.size()<2)throw std::runtime_error("update() braucht 2 Argumente");tensor_update(args[0].as_tensor(n->line),args[1].as_number(n->line));return SkullValue();}
        if(n->name=="zero_grad"){if(args.empty())throw std::runtime_error("zero_grad() braucht 1 Argument");args[0].as_tensor(n->line)->zero_grad();return SkullValue();}
        if(n->name=="shape"){if(args.empty())throw std::runtime_error("shape() braucht 1 Argument");auto t=args[0].as_tensor(n->line);std::cout<<t->rows<<"x"<<t->cols;return SkullValue();}
        if(n->name=="get_loss"){if(args.empty())throw std::runtime_error("get_loss() braucht 1 Argument");auto t=args[0].as_tensor(n->line);if(t->rows==1&&t->cols==1)return SkullValue(t->data[0]);throw std::runtime_error("get_loss(): nicht 1x1!");}
        if(n->name=="sqrt"){if(args.empty())throw std::runtime_error("sqrt() braucht 1 Argument");return SkullValue(std::sqrt(args[0].as_number(n->line)));}
        if(n->name=="abs"){if(args.empty())throw std::runtime_error("abs() braucht 1 Argument");return SkullValue(std::abs(args[0].as_number(n->line)));}
        if(n->name=="floor"){if(args.empty())throw std::runtime_error("floor() braucht 1 Argument");return SkullValue(std::floor(args[0].as_number(n->line)));}
        if(n->name=="ceil"){if(args.empty())throw std::runtime_error("ceil() braucht 1 Argument");return SkullValue(std::ceil(args[0].as_number(n->line)));}
        if(n->name=="round"){if(args.empty())throw std::runtime_error("round() braucht 1 Argument");return SkullValue(std::round(args[0].as_number(n->line)));}
        if(n->name=="pow"){if(args.size()<2)throw std::runtime_error("pow() braucht 2 Argumente");return SkullValue(std::pow(args[0].as_number(n->line),args[1].as_number(n->line)));}
        if(n->name=="min"){if(args.size()<2)throw std::runtime_error("min() braucht 2 Argumente");return SkullValue(std::min(args[0].as_number(n->line),args[1].as_number(n->line)));}
        if(n->name=="max"){if(args.size()<2)throw std::runtime_error("max() braucht 2 Argumente");return SkullValue(std::max(args[0].as_number(n->line),args[1].as_number(n->line)));}
        if(n->name=="str"){if(args.empty())throw std::runtime_error("str() braucht 1 Argument");return SkullValue(args[0].to_string());}

        auto it=functions.find(n->name);
        if(it!=functions.end()){
            const SkullFunction&fn=it->second;
            if(args.size()!=fn.params.size())throw std::runtime_error("'"+n->name+"' falsche Argumentanzahl");
            auto fn_env=std::make_shared<Environment>(fn.closure);
            for(size_t i=0;i<fn.params.size();++i)fn_env->define(fn.params[i],args[i]);
            try{for(const auto&s:*fn.body)exec_stmt(s.get(),fn_env);}
            catch(ReturnSignal&ret){return ret.value;}
            return SkullValue();
        }
        throw std::runtime_error("Zeile "+std::to_string(n->line)+": Unbekannte Funktion '"+n->name+"'");
    }

    void exec_stmt(const StmtNode*node,std::shared_ptr<Environment>env){
        if(!node)return;
        if(auto*n=dynamic_cast<const AssignStmt*>(node)){env->assign(n->name,eval_expr(n->value.get(),env));return;}
        if(auto*n=dynamic_cast<const ModelStmt*>(node)){models[n->name]=const_cast<ModelStmt*>(n);std::cout<<"[Skull] Modell '"<<n->name<<"' definiert\n";return;}

        // ---- TRAIN (v0.7.0: BPE + GPU + Multi-Format) ----
        if(auto*n=dynamic_cast<const TrainStmt*>(node)){
            TrainConfig cfg;
            auto model_it=models.find(n->model_name);
            if(model_it!=models.end()){
                for(const auto&f:model_it->second->fields){
                    SkullValue val=eval_expr(f.value.get(),env);
                    if(f.name=="dim")  cfg.dim  =(size_t)val.as_number();
                    if(f.name=="vocab")cfg.vocab=(size_t)val.as_number();
                }
            }
            for(const auto&f:n->fields){
                SkullValue val=eval_expr(f.value.get(),env);
                if(f.name=="data")      cfg.data_path  =val.text;
                if(f.name=="epochs")    cfg.epochs     =(int)val.as_number();
                if(f.name=="rate")      cfg.rate       =val.as_number();
                if(f.name=="batch")     cfg.batch      =(int)val.as_number();
                if(f.name=="dim")       cfg.dim        =(size_t)val.as_number();
                if(f.name=="bpe")       cfg.use_bpe    =val.is_truthy();
                if(f.name=="bpe_vocab") cfg.bpe_vocab  =(int)val.as_number();
                if(f.name=="gpu")       cfg.use_gpu    =val.is_truthy();
                if(f.name=="prefer_amd")cfg.prefer_amd =val.is_truthy();
            }
            skull_train(cfg);
            return;
        }

        // ---- GENERATE ----
        if(auto*n=dynamic_cast<const GenerateStmt*>(node)){
            GenerateConfig cfg;
            auto model_it=models.find(n->model_name);
            if(model_it!=models.end()){
                for(const auto&f:model_it->second->fields){
                    SkullValue val=eval_expr(f.value.get(),env);
                    if(f.name=="dim")  cfg.dim  =(size_t)val.as_number();
                    if(f.name=="vocab")cfg.vocab=(size_t)val.as_number();
                }
            }
            for(const auto&f:n->fields){
                SkullValue val=eval_expr(f.value.get(),env);
                if(f.name=="weights")    cfg.weights_path=val.text;
                if(f.name=="prompt")     cfg.prompt      =val.text;
                if(f.name=="tokens")     cfg.tokens      =(int)val.as_number();
                if(f.name=="temperature")cfg.temperature =val.as_number();
                if(f.name=="dim")        cfg.dim         =(size_t)val.as_number();
            }
            skull_generate(cfg);
            return;
        }

        if(auto*n=dynamic_cast<const FuncStmt*>(node)){
            SkullFunction fn;fn.name=n->name;fn.params=n->params;
            fn.body=const_cast<std::vector<std::unique_ptr<StmtNode>>*>(&n->body);fn.closure=env;
            functions[n->name]=std::move(fn);return;
        }
        if(auto*n=dynamic_cast<const ReturnStmt*>(node)){SkullValue val;if(n->value)val=eval_expr(n->value.get(),env);throw ReturnSignal{val};}
        if(auto*n=dynamic_cast<const IfStmt*>(node)){
            SkullValue cond=eval_expr(n->condition.get(),env);
            auto be=std::make_shared<Environment>(env);
            if(cond.is_truthy()){for(const auto&s:n->then_body)exec_stmt(s.get(),be);}
            else{for(const auto&s:n->else_body)exec_stmt(s.get(),be);}
            return;
        }
        if(auto*n=dynamic_cast<const ForStmt*>(node)){
            double start=eval_expr(n->start.get(),env).as_number(n->line);
            double end=eval_expr(n->end.get(),env).as_number(n->line);
            for(long long i=(long long)start;i<=(long long)end;++i){
                auto le=std::make_shared<Environment>(env);
                le->define(n->var_name,SkullValue((double)i));
                for(const auto&s:n->body)exec_stmt(s.get(),le);
            }
            return;
        }
        if(auto*n=dynamic_cast<const WhileStmt*>(node)){
            int guard=10000000;
            while(eval_expr(n->condition.get(),env).is_truthy()){
                auto le=std::make_shared<Environment>(env);
                for(const auto&s:n->body)exec_stmt(s.get(),le);
                if(--guard<=0)throw std::runtime_error("while-Schleife zu lang");
            }
            return;
        }
        if(auto*n=dynamic_cast<const ExprStmt*>(node)){eval_expr(n->expr.get(),env);return;}
        throw std::runtime_error("Zeile "+std::to_string(node->line)+": Unbekannter Statement-Typ");
    }

public:
    Interpreter(){global_env=std::make_shared<Environment>();}
    void run(const ProgramNode*program){
        for(const auto&stmt:program->statements)exec_stmt(stmt.get(),global_env);
    }
};
