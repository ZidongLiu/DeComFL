import numpy as np
from defense import run_defense

def test_no_attack():
    np.random.seed(42)
    g = np.random.normal(10, 1, size=20)
    recovered_g, mu, sigma2, pi, z = run_defense(g, num_iters=100)
    
    print("=== Test: No Attack ===")
    print("Original gradients:")
    print(g)
    print("Recovered gradients:")
    print(recovered_g)
    print("Estimated mu:", mu)
    print("Estimated sigma^2:", sigma2)
    print("Estimated π:", pi)
    print("z flags (0: unmodified, 1: modified):")
    print(z)
    
    assert np.sum(z) == 0, "Expected no modifications to be detected."

def test_attack():
    np.random.seed(42)
    g = np.random.normal(10, 1, size=20)
    g_modified = g.copy()
    
    g_modified[5] = 15
    g_modified[10] = -5
    g_modified[15] = 20
    
    recovered_g, mu, sigma2, pi, z = run_defense(g_modified, num_iters=100)
    
    print("\n=== Test: Attack ===")
    print("Original (modified) gradients:")
    print(g_modified)
    print("Recovered gradients:")
    print(recovered_g)
    print("Estimated mu:", mu)
    print("Estimated sigma^2:", sigma2)
    print("Estimated π:", pi)
    print("z flags (0: unmodified, 1: modified):")
    print(z)
    
    assert z[5] == 1 and z[10] == 1 and z[15] == 1, "Modified indices should be flagged as modified."

if __name__ == '__main__':
    test_no_attack()
    test_attack()
