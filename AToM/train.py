from args import parse_train_opt
from AToM import AToM


def train(opt):
    model = AToM(opt.feature_type, checkpoint_path = opt.checkpoint)
    model.train_loop(opt)

if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
