import argparse
import json
import sys
from pathlib import Path

from .habitat import (DEFAULT_CATEGORICAL_WEIGHT, DEFAULT_MIN_MARGIN, DEFAULT_SEED,
                      HabitatModel, aggregate_cells)
from .io import load_records, save_json
from .selection import select_k
from .summary import summarize


def main():
    parser = argparse.ArgumentParser(description='本地观测数据分析')
    commands = parser.add_subparsers(dest='command', required=True)
    summary = commands.add_parser('summary', help='按业务维度核对观测口径')
    summary.add_argument('--input', type=Path, default=Path('data/observations.csv'))
    summary.add_argument('--output', type=Path)
    habitat = commands.add_parser('habitat', help='训练或应用生境分群模型')
    habitat.add_argument('--input', type=Path, default=Path('data/observations.csv'))
    habitat.add_argument('--model', type=Path, required=True,
                         help='模型产物路径：训练时写出，--predict 时读入')
    habitat.add_argument('--assignments', type=Path, help='逐网格归属与置信诊断输出路径')
    habitat.add_argument('--predict', action='store_true',
                         help='载入既有模型进行预测，不重新训练')
    habitat.add_argument('--k', type=int, help='固定簇数，跳过自动选择')
    habitat.add_argument('--k-max', type=int, default=6, help='自动选择的最大候选簇数')
    habitat.add_argument('--seed', type=int, default=DEFAULT_SEED, help='训练随机种子')
    habitat.add_argument('--min-margin', type=float, default=DEFAULT_MIN_MARGIN,
                         help='置信边际阈值，低于则判为无可靠归属（训练时写入产物）')
    habitat.add_argument('--categorical-weight', type=float, default=DEFAULT_CATEGORICAL_WEIGHT,
                         help='底质编码块在特征向量中的权重')
    args = parser.parse_args()
    try:
        if args.command == 'summary':
            result = summarize(load_records(args.input))
            if args.output:
                save_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        elif args.predict:
            model = HabitatModel.load(args.model)
            result = model.predict(load_records(args.input))
            if args.assignments:
                save_json(args.assignments, result)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        else:
            records = load_records(args.input)
            if args.k is not None:
                k, selection = args.k, None
            else:
                k, selection = select_k(aggregate_cells(records), k_max=args.k_max,
                                        seed=args.seed,
                                        categorical_weight=args.categorical_weight)
            model = HabitatModel.train(records, k=k, selection=selection, seed=args.seed,
                                       categorical_weight=args.categorical_weight,
                                       min_margin=args.min_margin)
            model.save(args.model)
            if args.assignments:
                save_json(args.assignments, model.predict(records))
            print(json.dumps(model.artifact, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
