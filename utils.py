
import os

def get_result_path(args):
    result_path = os.getcwd() + "/results2/"
    result_path += args.t5_model.split("/")[-1]
    if args.prompt_tuning:
        result_path += "-pt" + str(args.prompt_length)
        if args.prompt_scaling:
            result_path += "s"
            if args.element_wise_scaling:
                result_path += "ew"
    
    result_path += "/seed" + str(args.seed)
    if args.precision == "32":
        result_path += "-fp32"
    result_path += "/"
    return result_path