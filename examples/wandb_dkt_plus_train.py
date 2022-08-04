import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    train_args = parser.add_argument_group('Train_args')

    train_args.add_argument("--dataset_name", type=str, default="assist2015")
    train_args.add_argument("--model_name", type=str, default="dkt+")
    train_args.add_argument("--emb_type", type=str, default="qid")
    train_args.add_argument("--save_dir", type=str, default="saved_model")
    train_args.add_argument("--seed", type=int, default=42)
    train_args.add_argument("--fold", type=int, default=5) #

    train_args.add_argument("--batch_size", type=int, default=256) #
    train_args.add_argument("--num_epochs", type=int, default=200) #
    train_args.add_argument("--seq_len", type=int, default=200) #
    train_args.add_argument("--es_patience", type=int, default=10) #
    train_args.add_argument("--gpu", type=str, default="0") #
    
    model_args = parser.add_argument_group('Model_args')

    model_args.add_argument("--dropout", type=float, default=0.2)
    model_args.add_argument("--emb_size", type=int, default=200)
    model_args.add_argument("--learning_rate", type=float, default=1e-3)
    model_args.add_argument("--lambda_r", type=float, default=0.01)
    model_args.add_argument("--lambda_w1", type=float, default=0.003)
    model_args.add_argument("--lambda_w2", type=float, default=3.0)
   
    model_args.add_argument("--use_wandb", type=int, default=1)
    model_args.add_argument("--add_uuid", type=int, default=1)

    args = parser.parse_args()
    
    arg_groups = {}
    for group in parser._action_groups:
        if group.title in ['positional arguments', 'optional arguments']: continue
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=vars(argparse.Namespace(**group_dict))

    train_params, model_params = arg_groups['Train_args'], arg_groups['Model_args']
    main(train_params, model_params)

