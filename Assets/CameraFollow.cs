using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    [Header("要追蹤的目標 (請放入 Leader 船)")]
    public Transform target;

    [Header("攝影機相對目標的偏移量 (X, Y, Z)")]
    public Vector3 offset = new Vector3(0f, 60f, -20f); // 預設高度 60，往後退 20

    // 使用 LateUpdate 而不是 Update，能避免畫面抖動
    void LateUpdate()
    {
        if (target != null)
        {
            // 讓攝影機的位置 = 目標的位置 + 偏移量 (完全不改動旋轉角度)
            transform.position = target.position + offset;
        }
    }
}